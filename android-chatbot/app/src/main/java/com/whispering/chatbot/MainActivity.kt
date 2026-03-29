package com.whispering.chatbot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(val role: String, val text: String)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0B0F16)) {
                    ChatScreen()
                }
            }
        }
    }
}

@Composable
fun ChatScreen() {
    val scope = rememberCoroutineScope()
    var input by remember { mutableStateOf("") }
    var loading by remember { mutableStateOf(false) }
    var modelReady by remember { mutableStateOf(false) }
    val messages = remember {
        mutableStateListOf(
            ChatMessage("system", "Whispering Shadows AI (SmolLM2-135M-Instruct local mode)")
        )
    }

    LaunchedEffect(Unit) {
        loading = true
        modelReady = withContext(Dispatchers.Default) {
            try {
                val py = Python.getInstance()
                val module = py.getModule("chatbot")
                module.callAttr("initialize_model")
                true
            } catch (e: Exception) {
                messages += ChatMessage("system", "Model init failed: ${e.message}")
                false
            }
        }
        loading = false
    }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        Text(
            text = "Whispering Shadows AI",
            color = Color(0xFFE2E8F0),
            fontWeight = FontWeight.Bold,
            fontSize = 22.sp
        )
        Text(
            text = if (modelReady) "Local model ready" else "Loading local model...",
            color = if (modelReady) Color(0xFF34D399) else Color(0xFFF59E0B),
            fontSize = 12.sp
        )

        Spacer(Modifier.height(8.dp))

        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { msg ->
                val isUser = msg.role == "user"
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
                ) {
                    Text(
                        text = msg.text,
                        modifier = Modifier
                            .widthIn(max = 320.dp)
                            .background(
                                if (isUser) Color(0xFF2563EB) else Color(0xFF1F2937),
                                RoundedCornerShape(14.dp)
                            )
                            .padding(10.dp),
                        color = Color.White
                    )
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                placeholder = { Text("Ask anything (offline)") },
                modifier = Modifier.weight(1f)
            )
            Button(
                enabled = input.isNotBlank() && !loading && modelReady,
                onClick = {
                    val prompt = input.trim()
                    input = ""
                    messages += ChatMessage("user", prompt)
                    loading = true

                    scope.launch {
                        val reply = withContext(Dispatchers.Default) {
                            try {
                                val py = Python.getInstance()
                                val module = py.getModule("chatbot")
                                val output: PyObject = module.callAttr("generate_response", prompt)
                                output.toString()
                            } catch (e: Exception) {
                                "Error: ${e.message}"
                            }
                        }
                        messages += ChatMessage("assistant", reply)
                        loading = false
                    }
                }
            ) {
                Text(if (loading) "..." else "Send")
            }
        }
    }
}
