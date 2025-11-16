package com.example.blindnavigator

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

class ModeActivity : AppCompatActivity() {

    private lateinit var ttsManager: TtsManager

    private lateinit var btnModeHelp: Button
    private lateinit var btnOutdoor: Button
    private lateinit var btnIndoor: Button
    private lateinit var btnModeVoice: Button

    companion object {
        private const val REQUEST_CODE_SPEECH_MODE = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_mode_select)

        ttsManager = TtsManager(this)

        btnModeHelp = findViewById(R.id.btnModeHelp)
        btnOutdoor = findViewById(R.id.btnOutdoor)
        btnIndoor = findViewById(R.id.btnIndoor)
        btnModeVoice = findViewById(R.id.btnModeVoice)

        // Один раз при первом запуске приложения рассказать про кнопку "Помощь"
        ttsManager.speakFirstLaunchIntroIfNeeded()

        // Кнопка "Помощь" – рассказывает про расположение кнопок на этом экране
        btnModeHelp.setOnClickListener {
            ttsManager.speakModeScreenHelp()
        }

        // Кнопка "Улица"
        btnOutdoor.setOnClickListener {
            val text = "Выбран режим улица"
            ttsManager.speakModeSelected(text)
            openNavigation("outdoor")
        }

        // Кнопка "Помещение"
        btnIndoor.setOnClickListener {
            val text = "Выбран режим помещение"
            ttsManager.speakModeSelected(text)
            openNavigation("indoor")
        }

        // Кнопка "Голос" – сразу прерывает всё и слушает
        btnModeVoice.setOnClickListener {
            ttsManager.stop()
            startSpeechRecognition()
        }
    }

    private fun startSpeechRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Скажите: улица, помещение или помощь")
        }

        try {
            startActivityForResult(intent, REQUEST_CODE_SPEECH_MODE)
        } catch (e: Exception) {
            ttsManager.speak("Распознавание речи недоступно на этом устройстве.", true)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_CODE_SPEECH_MODE && resultCode == Activity.RESULT_OK) {
            val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val phrase = results?.firstOrNull()?.lowercase(Locale.getDefault()) ?: return

            when {
                phrase.contains("улиц") -> {
                    val text = "Выбран режим улица"
                    ttsManager.speakModeSelected(text)
                    openNavigation("outdoor")
                }

                phrase.contains("помещ") || phrase.contains("комнат") || phrase.contains("дом") -> {
                    val text = "Выбран режим помещение"
                    ttsManager.speakModeSelected(text)
                    openNavigation("indoor")
                }

                phrase.contains("помощ") -> {
                    ttsManager.speakModeScreenHelp()
                }

                else -> {
                    ttsManager.speak(
                        "Я не поняла. Скажите: улица, помещение или помощь.",
                        true
                    )
                }
            }
        }
    }

    private fun openNavigation(mode: String) {
        val intent = Intent(this, NavigationActivity::class.java).apply {
            putExtra("mode", mode)
        }
        startActivity(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsManager.shutdown()
    }
}
