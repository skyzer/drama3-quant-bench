#!/bin/bash
# B2.5: wait sob A/B, pick winner by sob fail (tie -> emo_a), full 630 + extra sob seeds (88..154) on winner/fp8/w4a8, eval, analyze.
cd ~/AI/drama3-community; P=comfy/.venv/bin/python
while screen -ls | grep -q d3b2ab; do sleep 30; done
W=$(evalenv/bin/python - <<"Q"
import re
t=open("logs/b2_compare.txt").read().split("cats")[1]
f={m.group(1):float(m.group(2)) for m in re.finditer(r"^(b2_emo_\w)\s+fail=([\d.]+)",t,re.M)}
print(min(sorted(f),key=lambda k:f[k]))
Q
)
echo "$(date "+%F %T") B2.5 winner $W; full 630 start" >> PROGRESS.txt; echo $W > b2/WINNER
$P scripts/gen_pack.py b2_${W}_full models/drama3_${W#b2_}.safetensors --tag ${W}_full >> logs/b2_full.log 2>&1
echo "$(date "+%F %T") B2.5 gen ${W}_full done" >> PROGRESS.txt
for v in ${W}_full:models/drama3_${W#b2_}.safetensors pack_fp8:models/drama3_fp8.safetensors pack_w4a8:models/drama3_w4a8.safetensors; do
  D3_CATS=sob D3_SEEDS=88,99,110,121,132,143,154 $P scripts/gen_pack.py x_${v%%:*} ${v#*:} --tag xsob_${v%%:*} >> logs/b2_xsob.log 2>&1
done
echo "$(date "+%F %T") B2.5 extra sob seeds done" >> PROGRESS.txt
SP=$(evalenv/bin/python -c "import nvidia.cublas,os;print(os.path.dirname(nvidia.cublas.__path__[0]))"); export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
evalenv/bin/python scripts/eval.py ${W}_full xsob_${W}_full xsob_pack_fp8 xsob_pack_w4a8 >> logs/b2_eval.log 2>&1
evalenv/bin/python scripts/analyze.py > logs/analyze_b2_full.txt 2>&1
echo "$(date "+%F %T") B2.5 full scored" >> PROGRESS.txt
