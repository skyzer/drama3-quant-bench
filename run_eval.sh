#!/bin/bash
# A3 eval loop (screen d3eval): evaluates whatever takes exist, repeats until generation ends. Resumable.
cd ~/AI/drama3-community
SP=$(evalenv/bin/python -c "import nvidia.cublas,nvidia.cudnn,os;print(os.path.dirname(nvidia.cublas.__path__[0]))")
export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
while true; do
  T=$(ls out); evalenv/bin/python scripts/eval.py $T >> logs/eval.log 2>&1
  screen -ls | grep -qE "d3gen|d3extra" || { evalenv/bin/python scripts/eval.py $(ls out) >> logs/eval.log 2>&1; break; }
  sleep 120
done
echo "$(date "+%F %T") A3 eval loop finished" >> PROGRESS.txt
