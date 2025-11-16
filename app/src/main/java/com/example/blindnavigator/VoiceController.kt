package com.example.blindnavigator

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

class VoiceController(
    private val context: Context,
    private val onCommand: (String) -> Unit,
    private val onStatus: (String) -> Unit
) {

    private val tag = "VoiceController"

    private val recognizer: SpeechRecognizer =
        SpeechRecognizer.createSpeechRecognizer(context)

    private val recognizeIntent: Intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        // Свободная речь
        putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
        )
        // Русский язык
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        // Подсказка — можно и убрать
        putExtra(RecognizerIntent.EXTRA_PROMPT, "Говорите...")
    }

    init {
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Log.d(tag, "onReadyForSpeech")
                onStatus("Говорите…")
            }

            override fun onBeginningOfSpeech() {
                Log.d(tag, "onBeginningOfSpeech")
                onStatus("Слушаю…")
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Можно игнорировать или использовать для визуализации громкости
            }

            override fun onBufferReceived(buffer: ByteArray?) {
                // Не обязательно обрабатывать
            }

            override fun onEndOfSpeech() {
                Log.d(tag, "onEndOfSpeech")
                onStatus("Обработка…")
            }

            override fun onError(error: Int) {
                Log.d(tag, "onError: $error")
                val msg = when (error) {
                    SpeechRecognizer.ERROR_AUDIO -> "Ошибка аудио"
                    SpeechRecognizer.ERROR_CLIENT -> "Ошибка клиента"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Нет разрешения на микрофон"
                    SpeechRecognizer.ERROR_NETWORK -> "Ошибка сети"
                    SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Таймаут сети"
                    SpeechRecognizer.ERROR_NO_MATCH -> "Ничего не распознано"
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Распознавание уже идёт"
                    SpeechRecognizer.ERROR_SERVER -> "Ошибка сервера"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Вы не говорили"
                    else -> "Неизвестная ошибка ($error)"
                }
                onStatus("Голос: $msg")
                // При ошибке можно вернуть пустую строку как "нет команды"
                onCommand("")
            }

            override fun onResults(results: Bundle?) {
                Log.d(tag, "onResults: $results")
                val matches =
                    results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull().orEmpty()

                Log.d(tag, "onResults text='$text'")
                onStatus(
                    if (text.isBlank()) "Голос: ничего не распознано"
                    else "Голос: $text"
                )

                onCommand(text)
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val matches =
                    partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull().orEmpty()
                if (text.isNotBlank()) {
                    Log.d(tag, "onPartialResults: $matches")
                    onStatus("Слушаю: $text")
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                // Обычно не используется
            }
        })
    }

    /**
     * Запустить одноразовое распознавание.
     * MainActivity уже проверяет разрешения перед вызовом.
     */
    fun startOnce() {
        Log.d(tag, "startOnce()")
        onStatus("Запуск распознавания…")
        recognizer.startListening(recognizeIntent)
    }

    fun onDestroy() {
        Log.d(tag, "onDestroy()")
        recognizer.destroy()
    }
}
