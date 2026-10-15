#!/bin/bash
# A2 baseline. Frees ComfyUI VRAM, starts the production server binary unchanged (separate log), runs gen_fish, stops server.
cd ~/AI/drama3-community
curl -s -X POST http://127.0.0.1:8188/free -H 'content-type: application/json' -d '{"unload_models":true,"free_memory":true}'
pgrep -f "[a]pi_server.py.*8096" || (nohup ~/AI/tts/drama3/serve.sh > logs/fish_server.log 2>&1 &)
for i in $(seq 120); do curl -s -o /dev/null http://127.0.0.1:8096/v1/health && break; sleep 5; done
~/AI/tts/drama3/venv/bin/python scripts/gen_fish.py fish_fp8
pkill -f "[a]pi_server.py.*8096"
