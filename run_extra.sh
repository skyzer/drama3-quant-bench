#!/bin/bash
# Waits for d3gen, then fish_fp8 baseline, then GGUF. Skips sgl_int4 (Blackwell blocked).
cd ~/AI/drama3-community
while screen -ls | grep -q '\.d3gen'; do sleep 60; done
echo "$(date "+%F %T") A2b run_extra start (fish+gguf)" >> PROGRESS.txt
curl -s -X POST http://127.0.0.1:8188/free -H 'content-type: application/json' -d '{"unload_models":true,"free_memory":true}'
./run_fish.sh > logs/run_fish.log 2>&1
./run_gguf.sh > logs/run_gguf.log 2>&1
echo "$(date "+%F %T") A2b run_extra end" >> PROGRESS.txt
