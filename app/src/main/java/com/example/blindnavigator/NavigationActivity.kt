package com.example.blindnavigator

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.Locale

class NavigationActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvStatus: TextView
    private lateinit var btnWhatHere: Button
    private lateinit var btnToggle: Button
    private lateinit var btnVoiceControl: Button
    private lateinit var btnNavHelp: Button

    private lateinit var ttsManager: TtsManager
    private lateinit var navigationController: NavigationController

    companion object {
        private const val REQ_PERMISSIONS = 2001
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )

        private const val REQUEST_CODE_SPEECH_NAV = 2002
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigation)

        // --- Инициализация TTS ---
        ttsManager = TtsManager(this)

        // --- Находим view ---
        previewView = findViewById(R.id.previewView)
        tvStatus = findViewById(R.id.tvStatus)
        btnWhatHere = findViewById(R.id.btnWhatHere)
        btnToggle = findViewById(R.id.btnToggle)
        btnVoiceControl = findViewById(R.id.btnVoiceControl)
        btnNavHelp = findViewById(R.id.btnNavHelp)

        // --- Создаём NavigationController ---
        navigationController = NavigationController(
            activity = this,
            previewView = previewView,
            tvStatus = tvStatus,
            ttsManager = ttsManager
        )

        // Режим передаём из ModeActivity через intent.putExtra("mode", "outdoor" / "indoor")
        val modeStr = intent.getStringExtra("mode")
        navigationController.setModeFromString(modeStr)

        // --- Проверка разрешений ---
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQ_PERMISSIONS)
        } else {
            navigationController.startNavigation()
            btnToggle.text = "СТОП"
        }

        setupButtons()
    }

    private fun setupButtons() {
        // Кнопка "Помощь" сверху: поясняет расположение кнопок
        btnNavHelp.setOnClickListener {
            ttsManager.speakNavigationScreenHelp()
        }

        // "Что тут" – разовый анализ сцены (зелёная слева снизу)
        btnWhatHere.setOnClickListener {
            navigationController.handleWhatHere()
        }

        // "СТОП" – останавливаем навигацию и выходим назад (красная по центру снизу)
        btnToggle.setOnClickListener {
            navigationController.stopNavigation()
            finish()
        }

        // "ГОЛОС" – прерывает рассказ и слушает команды (жёлтая справа снизу)
        btnVoiceControl.setOnClickListener {
            ttsManager.stop()
            startSpeechRecognition()
        }
    }

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQ_PERMISSIONS) {
            if (allPermissionsGranted()) {
                navigationController.startNavigation()
                btnToggle.text = "СТОП"
            } else {
                ttsManager.speak(
                    "Без разрешения на камеру и микрофон навигация не работает.",
                    true
                )
            }
        }
    }

    private fun startSpeechRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
            putExtra(
                RecognizerIntent.EXTRA_PROMPT,
                "Скажите: стоп, что тут, помощь или начать"
            )
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_SPEECH_NAV)
        } catch (e: Exception) {
            ttsManager.speak("Распознавание речи недоступно на этом устройстве.", true)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_CODE_SPEECH_NAV && resultCode == Activity.RESULT_OK) {
            val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val phrase = results?.firstOrNull()?.lowercase(Locale.getDefault()) ?: return

            when {
                phrase.contains("стоп") -> {
                    navigationController.stopNavigation()
                    finish()
                }

                phrase.contains("что тут") ||
                        phrase.contains("что вокруг") ||
                        phrase.contains("что передо") ||
                        phrase.contains("опиши") -> {
                    navigationController.handleWhatHere()
                }

                phrase.contains("помощ") -> {
                    ttsManager.speakNavigationScreenHelp()
                }

                phrase.contains("начат") ||
                        phrase.contains("запуст") -> {
                    navigationController.startNavigation()
                    btnToggle.text = "СТОП"
                }

                else -> {
                    ttsManager.speak(
                        "Я не поняла. Скажите: стоп, что тут, помощь или начать.",
                        true
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        navigationController.onDestroy()
        ttsManager.shutdown()
    }
}
