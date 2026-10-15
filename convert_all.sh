#!/bin/bash
# A1: local conversions, resumable (skips finished files; convert.py writes .part then renames)
cd ~/AI/drama3-community
N=comfy/custom_nodes/ComfyUI-Drama3; SRC=~/AI/tts/drama3/checkpoints/drama-3; PY=comfy/.venv/bin/python
export PYTHONPATH=$PWD/comfy CUDA_VISIBLE_DEVICES=0
[ -f models/drama3_codec.safetensors ] || $PY $N/tools/convert.py codec $SRC/codec.pth models/drama3_codec.safetensors
for f in fp8 int8_convrot w6a8 w4a8; do
  o=models/drama3_${f}.safetensors
  [ -f $o ] && continue
  echo "$(date "+%F %T") A1 converting $f" >> PROGRESS.txt
  /usr/bin/time -v $PY $N/tools/convert.py model $SRC $o --format $f > logs/convert_$f.log 2>&1 && echo "$(date "+%F %T") A1 $f done $(du -h $o|cut -f1)" >> PROGRESS.txt || echo "$(date "+%F %T") A1 $f FAILED" >> PROGRESS.txt
done
