package com.example.blindnavigator

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okio.IOException
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

enum class NavigationMode(val apiName: String) {
    OUTDOOR("outdoor"),
    INDOOR("indoor")
}

class NavigationController(
    private val activity: ComponentActivity,
    private val previewView: PreviewView,
    private val tvStatus: TextView,
    private val ttsManager: TtsManager
) {

    companion object {
        private const val TAG = "NavigationController"
    }

    // подставь свой IP сервера
    private val serverBaseUrl = "http://195.209.210.250:8000"
    private val apiToken = "SUPER_SECRET_TOKEN_123"

    private var navRequestInProgress = false

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)   // чтобы выдержать даже редкие 10–20+ сек
        .build()
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val frameIntervalMs = 2000L   // каждые 2 секунды отправляем кадр

    // Порог дистанции для вибрации (метры)
    private val vibrateDistanceMinM = 0.2f
    private val vibrateDistanceMaxM = 1.2f

    // Задержки речи
    private val normalNavTtsDelayMs = 5000L   // 5 секунд между обычными фразами
    private val noObstacleTtsDelayMs = 10000L // 10 секунд между "препятствий нет"

    private var lastNavTtsTimeMs: Long = 0
    private var lastNavSentence: String? = null
    private var lastVibrateTimeMs: Long = 0

    private val vibrator: Vibrator?

    var isNavigationRunning: Boolean = false
        private set

    var navigationMode: NavigationMode = NavigationMode.OUTDOOR
        private set

    init {
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm =
                activity.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            activity.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    fun setMode(mode: NavigationMode) {
        navigationMode = mode
        val text = when (mode) {
            NavigationMode.OUTDOOR -> "Режим: улица"
            NavigationMode.INDOOR -> "Режим: помещение"
        }
        Log.d(TAG, "setMode: $mode, apiName=${mode.apiName}")
        activity.runOnUiThread {
            tvStatus.text = text
        }
        ttsManager.speak(text, true)
    }

    fun setModeFromString(modeStr: String?) {
        val m = when {
            modeStr.equals("indoor", ignoreCase = true) ||
                    modeStr.equals("INDOOR", ignoreCase = true) -> NavigationMode.INDOOR
            else -> NavigationMode.OUTDOOR
        }
        Log.d(TAG, "setModeFromString: '$modeStr' -> $m")
        setMode(m)
    }

    private fun buildAnalyzeUrl(): String {
        val modeParam = navigationMode.apiName
        return "$serverBaseUrl/analyze?mode=$modeParam&debug=1"
    }

    fun startNavigation() {
        if (isNavigationRunning) {
            Log.d(TAG, "startNavigation: already running, ignore")
            return
        }
        isNavigationRunning = true

        val url = buildAnalyzeUrl()
        Log.d(TAG, "startNavigation: mode=$navigationMode, url=$url")

        activity.runOnUiThread {
            tvStatus.text = "Навигация включена"
        }
        ttsManager.speak("Навигация включена.", true)

        startCameraIfNeeded()
        startAutoFrames()
    }

    fun stopNavigation() {
        if (!isNavigationRunning) {
            Log.d(TAG, "stopNavigation: already stopped, ignore")
            return
        }
        isNavigationRunning = false

        Log.d(TAG, "stopNavigation called")

        activity.runOnUiThread {
            tvStatus.text = "Навигация остановлена"
        }
        ttsManager.speak("Навигация остановлена.", true)

        stopAutoFrames()
        stopCamera()
    }

    fun startCameraIfNeeded() {
        Log.d(TAG, "startCameraIfNeeded")
        val providerFuture = ProcessCameraProvider.getInstance(activity)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                // Мы отправляем bitmap через previewView, поэтому тут просто освобождаем кадр
                imageProxy.close()
            }

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    activity,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                Log.d(TAG, "Camera bound to lifecycle")
            } catch (e: Exception) {
                Log.e(TAG, "Camera start error", e)
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun stopCamera() {
        try {
            Log.d(TAG, "stopCamera")
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "stopCamera error", e)
        }
    }

    // ===== Автосъёмка кадров для навигации =====

    private val autoFrameRunnable = object : Runnable {
        override fun run() {
            if (!isNavigationRunning) return

            val bitmap = previewView.bitmap
            if (bitmap != null && !navRequestInProgress) {
                sendFrameToServerNav(bitmap)
            }

            mainHandler.postDelayed(this, frameIntervalMs)
        }
    }

    private fun startAutoFrames() {
        Log.d(TAG, "startAutoFrames, interval=${frameIntervalMs}ms")
        mainHandler.removeCallbacks(autoFrameRunnable)
        mainHandler.post(autoFrameRunnable)
    }

    private fun stopAutoFrames() {
        Log.d(TAG, "stopAutoFrames")
        mainHandler.removeCallbacks(autoFrameRunnable)
    }

    // ===== Кнопка "Что тут?" =====

    fun handleWhatHere() {
        Log.d(TAG, "handleWhatHere called, isNavigationRunning=$isNavigationRunning")

        if (!isNavigationRunning) {
            ttsManager.speak(
                "Сначала включите навигацию. Для этого вернитесь назад и снова выберите режим.",
                true
            )
            return
        }

        val bitmap = previewView.bitmap
        if (bitmap == null) {
            Log.w(TAG, "handleWhatHere: bitmap is null – нет изображения с камеры")
            activity.runOnUiThread {
                tvStatus.text = "Нет изображения с камеры"
            }
            ttsManager.speak("Не вижу изображение с камеры.", true)
            return
        }

        Log.d(TAG, "handleWhatHere: got bitmap ${bitmap.width}x${bitmap.height}")

        activity.runOnUiThread {
            tvStatus.text = "Анализирую, подождите..."
        }
        ttsManager.speak("Анализирую сцену, подождите.", true)
        sendFrameToServerWhatHere(bitmap)
    }

    // ===== Отправка кадров на сервер =====

    private fun sendFrameToServerNav(bitmap: Bitmap) {
        try {
            navRequestInProgress = true  // <--- ставим флаг

            val bytes = bitmapToJpeg(bitmap)

            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    "frame.jpg",
                    bytes.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url(buildAnalyzeUrl())
                .addHeader("X-API-Key", apiToken)
                .post(body)
                .build()

            client.newCall(request).enqueue(object : okhttp3.Callback {
                override fun onFailure(call: okhttp3.Call, e: IOException) {
                    navRequestInProgress = false  // <--- сбрасываем
                    Log.e(TAG, "Request /analyze error", e)
                }

                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    navRequestInProgress = false  // <--- сбрасываем
                    response.use {
                        if (!it.isSuccessful) {
                            Log.e(
                                TAG,
                                "Response /analyze not successful: ${it.code}"
                            )
                            return
                        }
                        val bodyStr = it.body?.string() ?: return
                        handleAnalyzeResponse(bodyStr)
                    }
                }
            })
        } catch (e: Exception) {
            navRequestInProgress = false
            Log.e(TAG, "sendFrameToServerNav error", e)
        }
    }

    private fun sendFrameToServerWhatHere(bitmap: Bitmap) {
        try {
            val bytes = bitmapToJpeg(bitmap)
            val url = buildAnalyzeUrl()
            Log.d(
                TAG,
                "sendFrameToServerWhatHere: url=$url, bytes=${bytes.size}, mode=$navigationMode"
            )

            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    "what_here.jpg",
                    bytes.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url(url)
                .addHeader("X-API-Key", apiToken)
                .post(body)
                .build()

            client.newCall(request).enqueue(object : okhttp3.Callback {
                override fun onFailure(call: okhttp3.Call, e: IOException) {
                    Log.e(TAG, "Request /analyze what_here error", e)
                    activity.runOnUiThread {
                        tvStatus.text = "Ошибка анализа сцены"
                    }
                    ttsManager.speak("Не удалось получить описание сцены.", true)
                }

                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    response.use {
                        Log.d(TAG, "sendFrameToServerWhatHere: response code=${it.code}")
                        if (!it.isSuccessful) {
                            Log.e(
                                TAG,
                                "Response /analyze what_here not successful: ${it.code}"
                            )
                            activity.runOnUiThread {
                                tvStatus.text = "Ошибка анализа сцены"
                            }
                            ttsManager.speak("Ошибка анализа сцены.", true)
                            return
                        }
                        val bodyStr = it.body?.string() ?: return
                        Log.d(
                            TAG,
                            "sendFrameToServerWhatHere: response body length=${bodyStr.length}"
                        )
                        handleAnalyzeResponseWhatHere(bodyStr)
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "sendFrameToServerWhatHere error", e)
        }
    }

    // ===== Разбор ответов сервера (навигация) =====

    private fun handleAnalyzeResponse(jsonStr: String) {
        try {
            // Логируем сырой JSON для дебага
            Log.d(TAG, "handleAnalyzeResponse: raw json=$jsonStr")

            val json = JSONObject(jsonStr)

            // Универсальная проверка "ok" / "status"
            val ok = when {
                json.has("ok") -> {
                    when (val v = json.get("ok")) {
                        is Boolean -> v
                        is String -> v.equals("true", ignoreCase = true)
                        is Number -> v.toInt() != 0
                        else -> true
                    }
                }

                json.has("status") -> {
                    json.optString("status").equals("ok", ignoreCase = true)
                }

                else -> {
                    // Если сервер явно не сказал, что всё плохо — считаем, что ок
                    true
                }
            }

            if (!ok) {
                Log.w(TAG, "handleAnalyzeResponse: ok/status indicates error")
                return
            }

            // Возможные поля с описанием
            val rawDescription = json.optString("description")
            val rawText = json.optString("text")
            val rawSpeech = json.optString("speech")

            // Для статуса на экране
            val description = when {
                rawDescription.isNotBlank() -> rawDescription
                rawText.isNotBlank() -> rawText
                rawSpeech.isNotBlank() -> rawSpeech
                else -> ""
            }

            // Короткое описание, если есть
            val shortDescriptionField = json.optString("short_description")
            val shortDescription = when {
                shortDescriptionField.isNotBlank() -> shortDescriptionField
                else -> description
            }

            // Текст для озвучки: приоритет speech, потом short_description/description
            val speechText = when {
                rawSpeech.isNotBlank() -> rawSpeech
                shortDescription.isNotBlank() -> shortDescription
                description.isNotBlank() -> description
                else -> ""
            }

            val objectsArr: JSONArray? = json.optJSONArray("objects")
            var minDist: Float? = null

            if (objectsArr != null) {
                for (i in 0 until objectsArr.length()) {
                    val obj = objectsArr.optJSONObject(i) ?: continue
                    val d = obj.optDouble("distance_m", -1.0)
                    if (d > 0) {
                        val f = d.toFloat()
                        if (minDist == null || f < minDist!!) {
                            minDist = f
                        }
                    }
                }
            }

            val isDanger = minDist != null &&
                    minDist!! in vibrateDistanceMinM..vibrateDistanceMaxM

            activity.runOnUiThread {
                if (!isNavigationRunning) return@runOnUiThread

                // Что показать на экране
                val statusText = when {
                    description.isNotBlank() -> description
                    speechText.isNotBlank() -> speechText
                    else -> "Навигация активна"
                }
                tvStatus.text = statusText

                if (speechText.isNotBlank()) {
                    handleNavSpeech(speechText, isDanger)
                } else {
                    Log.d(TAG, "handleAnalyzeResponse: no text to speak")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "handleAnalyzeResponse error", e)
            ttsManager.speak("Ошибка обработки ответа сервера.", true)
        }
    }

    private fun handleNavSpeech(rawText: String, isDanger: Boolean) {
        val text = rawText.trim()
        if (text.isEmpty()) return

        val now = System.currentTimeMillis()
        val sameAsLast = (text == lastNavSentence)
        val isNoObstacles =
            text.contains("препятств", ignoreCase = true) &&
                    text.contains("нет", ignoreCase = true)

        val baseDelay = if (isNoObstacles) noObstacleTtsDelayMs else normalNavTtsDelayMs
        val enoughTimePassed = now - lastNavTtsTimeMs > baseDelay

        Log.d(
            TAG,
            "handleNavSpeech: text='$text', isDanger=$isDanger, sameAsLast=$sameAsLast, enoughTimePassed=$enoughTimePassed"
        )

        if (isDanger) {
            // Опасность — всегда говорим сразу
            ttsManager.speak(text, true)
            lastNavSentence = text
            lastNavTtsTimeMs = now
        } else {
            // Не опасность — не спамим одинаковыми фразами
            if (!sameAsLast && enoughTimePassed) {
                ttsManager.speak(text, true)
                lastNavSentence = text
                lastNavTtsTimeMs = now
            }
        }
    }

    // ===== Разбор ответа для "что тут" =====

    private fun handleAnalyzeResponseWhatHere(jsonStr: String) {
        try {
            Log.d(TAG, "handleAnalyzeResponseWhatHere: raw json=$jsonStr")

            val json = JSONObject(jsonStr)

            val ok = when {
                json.has("ok") -> {
                    when (val v = json.get("ok")) {
                        is Boolean -> v
                        is String -> v.equals("true", ignoreCase = true)
                        is Number -> v.toInt() != 0
                        else -> true
                    }
                }

                json.has("status") -> {
                    json.optString("status").equals("ok", ignoreCase = true)
                }

                else -> true
            }

            if (!ok) {
                Log.w(TAG, "handleAnalyzeResponseWhatHere: ok/status indicates error")
                activity.runOnUiThread {
                    tvStatus.text = "Ошибка анализа сцены"
                }
                ttsManager.speak("Ошибка анализа сцены.", true)
                return
            }

            val rawDescription = json.optString("description")
            val rawText = json.optString("text")
            val rawSpeech = json.optString("speech")
            val rawSceneTts = json.optString("scene_tts")
            val shortDescriptionField = json.optString("short_description")

            val description = when {
                rawDescription.isNotBlank() -> rawDescription
                rawText.isNotBlank() -> rawText
                rawSpeech.isNotBlank() -> rawSpeech
                else -> ""
            }

            val shortDescription = when {
                shortDescriptionField.isNotBlank() -> shortDescriptionField
                description.isNotBlank() -> description
                else -> ""
            }

            val sceneTts = when {
                rawSceneTts.isNotBlank() -> rawSceneTts
                rawSpeech.isNotBlank() -> rawSpeech
                shortDescription.isNotBlank() -> shortDescription
                description.isNotBlank() -> description
                else -> ""
            }

            val text = if (sceneTts.isNotBlank()) {
                sceneTts
            } else {
                "Пока нет описания сцены."
            }

            activity.runOnUiThread {
                val statusText = if (description.isNotBlank()) description else text
                tvStatus.text = statusText
            }
            ttsManager.speak(text, true)

        } catch (e: Exception) {
            Log.e(TAG, "handleAnalyzeResponseWhatHere error", e)
            activity.runOnUiThread {
                tvStatus.text = "Ошибка анализа сцены"
            }
            ttsManager.speak("Ошибка анализа сцены.", true)
        }
    }

    // ===== Вспомогательные =====

    private fun bitmapToJpeg(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
        return stream.toByteArray()
    }

    private fun vibrate() {
        val v = vibrator ?: return
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val effect = VibrationEffect.createOneShot(
                    150,
                    VibrationEffect.DEFAULT_AMPLITUDE
                )
                v.vibrate(effect)
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(150)
            }
        } catch (e: Exception) {
            Log.e(TAG, "vibrate error", e)
        }
    }

    fun onDestroy() {
        Log.d(TAG, "onDestroy")
        stopAutoFrames()
        stopCamera()
        cameraExecutor.shutdown()
    }
}
