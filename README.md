# Whispering Shadows AI (Android Local Chatbot)

This repository now includes an **Android chatbot app** which runs **HuggingFaceTB/SmolLM2-135M-Instruct locally on-device**.

## What was added

- Native Android app in `android-chatbot/` (Jetpack Compose UI).
- Local Python inference bridge via Chaquopy (`android-chatbot/app/src/main/python/chatbot.py`).
- Model expected in app assets at:
  - `app/src/main/assets/models/HuggingFaceTB/SmolLM2-135M-Instruct`
- GitHub Action to download model safetensors and build APK:
  - `.github/workflows/build-chatbot-apk.yml`
- Local helper script:
  - `scripts/build_chatbot_apk.sh`

## Build APK in GitHub Actions (recommended)

1. Push to GitHub.
2. Open **Actions**.
3. Run **Build Local AI Chatbot APK**.
4. Download artifact `whispering-shadows-ai-debug-apk`.

## Local build

```bash
./scripts/build_chatbot_apk.sh
```

APK output:

```text
android-chatbot/app/build/outputs/apk/debug/app-debug.apk
```

## Notes

- This flow downloads model files (including safetensors) from Hugging Face during build and packages them into the APK assets.
- Model packaging can significantly increase APK size.
