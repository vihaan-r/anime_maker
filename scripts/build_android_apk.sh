#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Phantom Lockdown"
APP_ID="com.phantom.lockdown"
WEB_DIR="www"

if ! command -v node >/dev/null 2>&1; then
  echo "❌ Node.js is required (v20+ recommended)." >&2
  exit 1
fi

if [ ! -f package.json ]; then
  npm init -y >/dev/null
fi

npm install @capacitor/core @capacitor/cli @capacitor/android --save-dev

# Capacitor does not accept '.' as webDir. Build a dedicated web bundle folder.
rm -rf "$WEB_DIR"
mkdir -p "$WEB_DIR"
cp index.html "$WEB_DIR"/
cp game.js "$WEB_DIR"/
cp styles.css "$WEB_DIR"/
if [ -d assets ]; then
  cp -R assets "$WEB_DIR"/
fi

cat > capacitor.config.json <<CONFIG
{
  "appId": "$APP_ID",
  "appName": "$APP_NAME",
  "webDir": "$WEB_DIR"
}
CONFIG

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
