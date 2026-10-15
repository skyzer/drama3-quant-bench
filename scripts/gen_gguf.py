"""A2 GGUF Q4_K (Jmica, audio.cpp drama-3-native @4dbb4d1) via audiocpp_server on :8035. Resumable. usage: gen_gguf.py [TAG]"""
import sys, os, json, time, requests, soundfile as sf, subprocess
ROOT = os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0, ROOT + "/scripts")
from lines import LINES, VOICES, SEEDS, TEMP, TOP_P
tag = sys.argv[1] if len(sys.argv) > 1 else "gguf_q4k"; OUT = f"{ROOT}/out/{tag}"; os.makedirs(OUT, exist_ok=True)
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
def gpu_used():
    try: return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0]) / 1024
    except Exception: return None
meta = open(OUT + "/meta.jsonl", "a"); n = 0
for lid, cat, text in LINES:
    for v in VOICES:
        for sd in SEEDS:
            p = f"{OUT}/{v}_{lid}_s{sd}.wav"
            if os.path.exists(p): continue
            opts = {"temperature": TEMP, "top_p": TOP_P, "top_k": 30, "seed": sd, "max_tokens": 600}
            req = {"model": "drama-3", "input": text, "voice": v, "response_format": "wav", **opts, "request_options": {k: str(x) for k, x in opts.items()}}
            t0 = time.time(); r = requests.post("http://127.0.0.1:8035/v1/audio/speech", json=req, timeout=900); gen_s = time.time() - t0
            if not r.ok: print("ERR", lid, v, sd, r.status_code, r.text[:300], flush=True); continue
            open(p + ".tmp.wav", "wb").write(r.content); os.replace(p + ".tmp.wav", p)
            info = sf.info(p); dur = info.frames / info.samplerate
            rec = dict(variant=tag, tag=tag, voice=v, line=lid, cat=cat, seed=sd, frames=int(round(dur * 21.533)), dur=round(dur, 3), gen_s=round(gen_s, 3),
                       rtf=round(gen_s / max(dur, 1e-3), 3), stall=None, hit_max=dur >= 600 / 21.533 - 0.1, peak_vram_gb=gpu_used(), maxrss_gb=None, load_s=None, ram_cache=None,
                       note="audio.cpp server, all experts on GPU; rtf incl HTTP")
            meta.write(json.dumps(rec) + "\n"); meta.flush(); print(rec, flush=True); n += 1
            if lim and n >= lim: sys.exit(0)
open(f"{ROOT}/PROGRESS.txt", "a").write(time.strftime("%F %T") + f" A2 gen {tag} complete\n")
