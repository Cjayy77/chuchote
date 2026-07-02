#!/usr/bin/env bash
# Install Chuchote, download a default Piper voice, and write a config file.
# Run from the repo root:  ./scripts/install.sh
set -euo pipefail

echo "Installing chuchote..."
python -m pip install -e .

echo "Locating voice directory..."
voice_dir="$(python -c 'from chuchote.config import user_voice_dir; import os; d=user_voice_dir(); os.makedirs(d, exist_ok=True); print(d)')"

base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
onnx="$voice_dir/en_US-lessac-medium.onnx"
json="$voice_dir/en_US-lessac-medium.onnx.json"
if [ ! -f "$onnx" ]; then
  echo "Downloading Piper voice (~63 MB)..."
  curl -sL -o "$onnx" "$base/en_US-lessac-medium.onnx"
  curl -sL -o "$json" "$base/en_US-lessac-medium.onnx.json"
else
  echo "Voice already present, skipping download."
fi

echo "Writing default config..."
python -m chuchote init || true

cat <<'EOF'

Done! Next:
  1. Start Ollama:    ollama serve
  2. Pull a model:    ollama pull llama3.2
  3. Talk to it:      chuchote start
EOF
