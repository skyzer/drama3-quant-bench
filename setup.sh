#!/bin/bash
# Phase A0 setup, idempotent. Separate ComfyUI checkout so production ComfyUI (0.37) is untouched.
set -x
cd ~/AI/drama3-community
P=PROGRESS.txt
[ -d comfy ] || git clone -q https://github.com/comfyanonymous/ComfyUI comfy
[ -d comfy/custom_nodes/ComfyUI-Drama3 ] || git clone -q https://github.com/PedroMarinhoDev/ComfyUI-Drama3 comfy/custom_nodes/ComfyUI-Drama3
cd comfy; git log -1 --format="%h %ci"; cat comfyui_version.py | tail -1
[ -x .venv/bin/python ] || ~/.local/bin/uv venv -q -p 3.12 .venv
~/.local/bin/uv pip install -q -p .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu130
~/.local/bin/uv pip install -q -p .venv/bin/python -r requirements.txt
.venv/bin/python -c "import torch,comfy_kitchen;print(torch.__version__,torch.cuda.is_available(),comfy_kitchen.__version__)"
echo "$(date "+%F %T") A0 setup done" >> ../$P
