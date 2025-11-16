package com.example.blindnavigator

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Сейчас не используется как главный экран.
 * В манифесте LAUNCHER стоит на ModeActivity.
 * Оставляем, чтобы проект собирался без ошибок.
 */
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Ничего не делаем
        finish()
    }
}
