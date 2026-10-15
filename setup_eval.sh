#!/bin/bash
cd ~/AI/drama3-community; U=~/.local/bin/uv
[ -x evalenv/bin/python ] || $U venv -q -p 3.12 evalenv
$U pip install -q -p evalenv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu128
$U pip install -q -p evalenv/bin/python faster-whisper speechbrain transformers librosa jiwer openai-whisper pandas matplotlib soundfile "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
echo "$(date "+%F %T") A3 eval env ready" >> PROGRESS.txt
