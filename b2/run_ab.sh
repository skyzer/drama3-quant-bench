#!/bin/bash
# B2.4 sob-first A/B: waits for builds, gens sob(+soblaugh) for each emo variant, evals, compares. Resumable.
cd ~/AI/drama3-community; P=comfy/.venv/bin/python
while screen -ls | grep -q d3b2b; do sleep 30; done
for v in emo_a emo_b; do
  [ -f models/drama3_$v.safetensors ] || continue
  D3_CATS=sob,soblaugh $P scripts/gen_pack.py b2_$v models/drama3_$v.safetensors --tag b2_$v >> logs/b2_gen_$v.log 2>&1
  echo "$(date "+%F %T") B2.4 gen b2_$v sob+soblaugh done" >> PROGRESS.txt
done
SP=$(evalenv/bin/python -c "import nvidia.cublas,os;print(os.path.dirname(nvidia.cublas.__path__[0]))"); export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
evalenv/bin/python scripts/eval.py b2_emo_a b2_emo_b >> logs/b2_eval.log 2>&1
for c in sob soblaugh; do evalenv/bin/python b2/compare.py $c pack_fp8 pack_w4a8 pack_w6a8 b2_emo_a b2_emo_b; done > logs/b2_compare.txt 2>&1
echo "$(date "+%F %T") B2.4 sob A/B scored -> logs/b2_compare.txt" >> PROGRESS.txt
