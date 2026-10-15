#!/bin/bash
# A2 pack variants, sequential, resumable. Quality runs: experts in VRAM. Speed/memory profile: experts in RAM w/ 8 GB cache, subset.
cd ~/AI/drama3-community
pkill -f "[a]pi_server.py.*8096"
curl -s -X POST http://127.0.0.1:8188/free -H 'content-type: application/json' -d '{"unload_models":true,"free_memory":true}'
PY=comfy/.venv/bin/python
for v in ${@:-fp8 int8_convrot w6a8 w4a8}; do
  $PY scripts/gen_pack.py pack_$v models/drama3_$v.safetensors --tag pack_$v >> logs/gen_pack_$v.log 2>&1
  $PY scripts/gen_pack.py pack_$v models/drama3_$v.safetensors --ram 8 --subset 6 --tag prof_ram8_$v >> logs/gen_prof_$v.log 2>&1
done
