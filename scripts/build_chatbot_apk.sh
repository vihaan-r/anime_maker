#!/usr/bin/env bash
set -euo pipefail

ROOT="android-chatbot"
MODEL_DIR="$ROOT/app/src/main/assets/models/HuggingFaceTB/SmolLM2-135M-Instruct"

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
command -v gradle >/dev/null || { echo "gradle required"; exit 1; }

python3 -m pip install --quiet --upgrade huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    local_dir="android-chatbot/app/src/main/assets/models/HuggingFaceTB/SmolLM2-135M-Instruct",
    local_dir_use_symlinks=False,
)
print("Model download complete")
PY

(
  cd "$ROOT"
  gradle :app:assembleDebug
)

echo "APK: $ROOT/app/build/outputs/apk/debug/app-debug.apk"
