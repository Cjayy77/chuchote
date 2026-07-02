#!/usr/bin/env pwsh
# Install Chuchote, download a default Piper voice, and write a config file.
# Run from the repo root:  ./scripts/install.ps1
$ErrorActionPreference = "Stop"

Write-Host "Installing chuchote..."
python -m pip install -e .

Write-Host "Locating voice directory..."
$voiceDir = (python -c "from chuchote.config import user_voice_dir; import os; d=user_voice_dir(); os.makedirs(d, exist_ok=True); print(d)").Trim()

$base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
$onnx = Join-Path $voiceDir "en_US-lessac-medium.onnx"
$json = Join-Path $voiceDir "en_US-lessac-medium.onnx.json"
if (-not (Test-Path $onnx)) {
    Write-Host "Downloading Piper voice (~63 MB)..."
    Invoke-WebRequest "$base/en_US-lessac-medium.onnx" -OutFile $onnx
    Invoke-WebRequest "$base/en_US-lessac-medium.onnx.json" -OutFile $json
} else {
    Write-Host "Voice already present, skipping download."
}

Write-Host "Writing default config..."
python -m chuchote init

Write-Host ""
Write-Host "Done! Next:"
Write-Host "  1. Start Ollama:    ollama serve"
Write-Host "  2. Pull a model:    ollama pull llama3.2"
Write-Host "  3. Talk to it:      chuchote start"
