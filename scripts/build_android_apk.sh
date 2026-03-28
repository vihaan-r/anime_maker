#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Phantom Lockdown"
APP_ID="com.phantom.lockdown"

if ! command -v node >/dev/null 2>&1; then
  echo "❌ Node.js is required (v20+ recommended)." >&2
  exit 1
fi

if [ ! -f package.json ]; then
  npm init -y >/dev/null
fi

npm install @capacitor/core @capacitor/cli @capacitor/android --save-dev

if [ ! -f capacitor.config.ts ] && [ ! -f capacitor.config.json ] && [ ! -f capacitor.config.js ]; then
  npx cap init "$APP_NAME" "$APP_ID" --web-dir .
fi

if [ ! -d android ]; then
  npx cap add android
fi

npx cap sync android

if [ -x android/gradlew ]; then
  (
    cd android
    ./gradlew assembleDebug
  )
  echo "✅ Debug APK built at: android/app/build/outputs/apk/debug/app-debug.apk"
else
  echo "⚠️ Android project created. Open it in Android Studio and build APK from there."
  echo "   Command: npx cap open android"
fi
