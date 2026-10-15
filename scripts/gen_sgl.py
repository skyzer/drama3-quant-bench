"""A2 SGLang-Omni variants (int4 GPTQ-Marlin and FP8 on the same runtime). Resumable.
usage: gen_sgl.py TAG PORT ; the server is started by run_extra.sh"""
import sys, os, json, time, requests, soundfile as sf, subprocess
ROOT = os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0, ROOT + "/scripts")
from lines import LINES, VOICES, SEEDS, TEMP, TOP_P
tag, port = sys.argv[1], sys.argv[2]; OUT = f"{ROOT}/out/{tag}"; os.makedirs(OUT, exist_ok=True)
refs = {v: (f"{ROOT}/testset/{v}_ref.wav", open(f"{ROOT}/testset/{v}_ref.txt").read().strip()) for v in VOICES}
def gpu_used():
    try: return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0]) / 1024
    except Exception: return None
meta = open(OUT + "/meta.jsonl", "a")
for lid, cat, text in LINES:
    for v in VOICES:
        for sd in SEEDS:
            p = f"{OUT}/{v}_{lid}_s{sd}.wav"
            if os.path.exists(p): continue
            w, t = refs[v]
            req = {"input": text, "references": [{"audio_path": w, "text": t}], "temperature": TEMP, "top_p": TOP_P, "top_k": 30,
                   "seed": sd, "max_new_tokens": 600, "response_format": "wav"}
            t0 = time.time(); r = requests.post(f"http://127.0.0.1:{port}/v1/audio/speech", json=req, timeout=900); gen_s = time.time() - t0
            if not r.ok: print("ERR", lid, v, sd, r.status_code, r.text[:300], flush=True); continue
            open(p + ".tmp.wav", "wb").write(r.content); os.replace(p + ".tmp.wav", p)
            info = sf.info(p); dur = info.frames / info.samplerate
            rec = dict(variant=tag, tag=tag, voice=v, line=lid, cat=cat, seed=sd, frames=int(round(dur * 21.533)), dur=round(dur, 3), gen_s=round(gen_s, 3),
                       rtf=round(gen_s / max(dur, 1e-3), 3), stall=None, hit_max=dur >= 600 / 21.533 - 0.1, peak_vram_gb=gpu_used(), maxrss_gb=None,
                       load_s=None, ram_cache=None, note="rtf incl. HTTP + ref encode; peak_vram = whole-GPU used (server pre-allocates)")
            meta.write(json.dumps(rec) + "\n"); meta.flush(); print(rec, flush=True)
open(f"{ROOT}/PROGRESS.txt", "a").write(time.strftime("%F %T") + f" A2 gen {tag} complete\n")
