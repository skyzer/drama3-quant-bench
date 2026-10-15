#!/bin/bash
# B2.5b extra sob seeds, experts in RAM (GPU shared with podcast H3 job). Resumable.
cd ~/AI/drama3-community; P=comfy/.venv/bin/python
for v in pack_fp8:models/drama3_fp8.safetensors:--ram=6 pack_w4a8:models/drama3_w4a8.safetensors:--ram=6 b2_emo_a_full:models/drama3_emo_a.safetensors:; do
  T=${v%%:*}; R=${v##*:}; M=${v#*:}; M=${M%:*}
  for try in $(seq 1 40); do
    D3_CATS=sob D3_SEEDS=88,99,110,121,132,143,154 $P scripts/gen_pack.py x_$T $M $R --tag xsob_$T >> logs/b2_xsob.log 2>&1; v=$T:x
    [ $(wc -l < out/xsob_${v%%:*}/meta.jsonl) -ge 84 ] && break; sleep 180
  done
done
echo "$(date "+%F %T") B2.5b extra sob seeds done (ram6)" >> PROGRESS.txt
SP=$(evalenv/bin/python -c "import nvidia.cublas,os;print(os.path.dirname(nvidia.cublas.__path__[0]))"); export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
evalenv/bin/python scripts/eval.py b2_emo_a_full xsob_b2_emo_a_full xsob_pack_fp8 xsob_pack_w4a8 >> logs/b2_eval.log 2>&1
evalenv/bin/python scripts/analyze.py > logs/analyze_b2_full.txt 2>&1
evalenv/bin/python - > logs/b2_xsob_compare.txt 2>&1 <<"Q"
import subprocess,sys
Q
echo "$(date "+%F %T") B2.5b scored" >> PROGRESS.txt
