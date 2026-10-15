cd ~/AI/drama3-community; P=comfy/.venv/bin/python
for t in 1 2 3 4 5; do $P scripts/gen_pack.py b2_emo_a_full models/drama3_emo_a.safetensors --tag b2_emo_a_full >> logs/b2_full.log 2>&1; [ $(ls out/b2_emo_a_full/*.wav | wc -l) -ge 630 ] && break; sleep 120; done
SP=$(evalenv/bin/python -c "import nvidia.cublas,os;print(os.path.dirname(nvidia.cublas.__path__[0]))"); export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cudnn/lib:$LD_LIBRARY_PATH
evalenv/bin/python scripts/eval.py b2_emo_a_full >> logs/b2_eval.log 2>&1
evalenv/bin/python scripts/analyze.py > logs/analyze_b2_full.txt 2>&1; evalenv/bin/python scripts/charts_cat.py >> logs/analyze_b2_full.txt 2>&1
echo "$(date "+%F %T") B2.6 emo_a full complete+scored, charts refreshed" >> PROGRESS.txt
