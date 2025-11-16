package com.example.blindnavigator

import android.content.Context
import android.content.SharedPreferences
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class TtsManager(private val context: Context) : TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "TtsManager"
        private const val PREFS_NAME = "tts_prefs"
        private const val KEY_FIRST_LAUNCH_DONE = "first_launch_done"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private var tts: TextToSpeech? = null
    private var isReady = false

    // Очередь фраз, если TTS ещё не инициализировался
    private val pendingQueue = mutableListOf<Pair<String, Boolean>>() // text, flush

    init {
        tts = TextToSpeech(context.applicationContext, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            try {
                val result = tts?.setLanguage(Locale("ru", "RU"))
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(TAG, "Русский язык не поддерживается, пробую Locale.getDefault()")
                    tts?.language = Locale.getDefault()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Ошибка установки языка TTS", e)
            }

            isReady = true

            // воспроизводим всё, что накопили до инициализации
            if (pendingQueue.isNotEmpty()) {
                for ((text, flush) in pendingQueue) {
                    internalSpeak(text, flush)
                }
                pendingQueue.clear()
            }
        } else {
            Log.e(TAG, "TTS init failed: status = $status")
        }
    }

    // Базовый метод
    fun speak(text: String, flush: Boolean = true) {
        if (text.isBlank()) return

        if (!isReady) {
            // TTS ещё не готов – добавляем в очередь
            pendingQueue += text to flush
            return
        }
        internalSpeak(text, flush)
    }

    private fun internalSpeak(text: String, flush: Boolean) {
        try {
            if (flush) {
                tts?.stop()
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts-${System.currentTimeMillis()}")
            } else {
                tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "tts-${System.currentTimeMillis()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Ошибка TTS при воспроизведении текста: $text", e)
        }
    }

    fun stop() {
        try {
            tts?.stop()
        } catch (_: Exception) {
        }
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Exception) {
        }
    }

    // ================= Дополнительные фразы =================

    /**
     * Сообщение только при первом запуске приложения.
     * Говорим, что везде есть большая красная кнопка помощи.
     */
    fun speakFirstLaunchIntroIfNeeded() {
        val already = prefs.getBoolean(KEY_FIRST_LAUNCH_DONE, false)
        if (already) return

        val text = """
            Добро пожаловать в Vision Guide.
            На каждом экране в верхней части есть большая красная кнопка Помощь.
            Нажмите её, чтобы услышать описание расположения кнопок и подсказки.
        """.trimIndent()

        speak(text, true)
        prefs.edit().putBoolean(KEY_FIRST_LAUNCH_DONE, true).apply()
    }

    /**
     * Помощь на экране выбора режима.
     *  - сверху красная кнопка Помощь,
     *  - по центру фиолетовая кнопка Помещение,
     *  - над ней кнопка Улица,
     *  - под ней жёлтая кнопка Голос.
     */
    fun speakModeScreenHelp() {
        val text = """
            Экран выбора режима.
            Вверху по всей ширине красная кнопка Помощь.
            По центру экрана фиолетовая кнопка Помещение.
            Выше неё расположена кнопка Улица.
            Ниже по центру жёлтая кнопка Голос.
            Нажмите Голос, чтобы управлять приложением голосом.
            При нажатии на Голос текущий рассказ остановится, и я начну слушать ваши команды.
        """.trimIndent()

        speak(text, true)
    }

    /**
     * Фраза после выбора режима: просто обёртка над speak, чтобы код в Activity был читабельнее.
     */
    fun speakModeSelected(text: String) {
        speak(text, true)
    }

    /**
     * Помощь на экране навигации.
     *  - сверху красная кнопка Помощь,
     *  - снизу по центру красная кнопка Стоп,
     *  - снизу слева зелёная кнопка Что тут,
     *  - снизу справа жёлтая кнопка Голос.
     */
    fun speakNavigationScreenHelp() {
        val text = """
            Экран навигации.
            Вверху по всей ширине красная кнопка Помощь.
            Внизу по центру красная кнопка Стоп, она завершает навигацию и возвращает вас назад.
            Внизу слева зелёная кнопка Что тут. Она разово описывает, что находится вокруг и перед вами.
            Внизу справа жёлтая кнопка Голос. При нажатии на Голос я перестану говорить и начну слушать команды.
            Вы можете сказать: стоп, что тут, помощь или начать.
        """.trimIndent()

        speak(text, true)
    }
}
