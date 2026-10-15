#!/bin/bash
# A2 orchestrator (screen d3gen). Resumable: rerun = continue. Order: pack variants first (fast feedback), then fish baseline.
cd ~/AI/drama3-community
echo "$(date "+%F %T") A2 run_all start" >> PROGRESS.txt
./run_pack.sh fp8 int8_convrot w6a8 w4a8
./run_fish.sh > logs/run_fish.log 2>&1
echo "$(date "+%F %T") A2 run_all end" >> PROGRESS.txt
