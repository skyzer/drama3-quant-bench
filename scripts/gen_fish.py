"""A2 baseline: official fish-speech api_server (FP8 experts as Fish ships them) on 127.0.0.1:8096. Resumable.
usage: gen_fish.py [TAG]  (server must be running; started by run_fish.sh)"""
import sys, os, json, time, requests, ormsgpack, soundfile as sf, subprocess
ROOT = os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0, ROOT + "/scripts")
from lines import LINES, VOICES, SEEDS, TEMP, TOP_P
tag = sys.argv[1] if len(sys.argv) > 1 else "fish_fp8"; OUT = f"{ROOT}/out/{tag}"; os.makedirs(OUT, exist_ok=True)
refs = {v: (open(f"{ROOT}/testset/{v}_ref.wav", "rb").read(), open(f"{ROOT}/testset/{v}_ref.txt").read().strip()) for v in VOICES}
def vram():
    try: return float(subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout and
                     max(int(l.split(",")[1]) for l in subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip().splitlines() if l.split(",")[0].strip()==PID))/1024
    except Exception: return None
PID = subprocess.run(["pgrep","-f","[a]pi_server.py.*8096"],capture_output=True,text=True).stdout.split()[0]
meta = open(OUT + "/meta.jsonl", "a")
for lid, cat, text in LINES:
    for v in VOICES:
        for sd in SEEDS:
            p = f"{OUT}/{v}_{lid}_s{sd}.wav"
            if os.path.exists(p): continue
            w, t = refs[v]
            req = {"text": text, "references": [{"audio": w, "text": t}], "format": "wav", "normalize": False, "single_chunk": True,
                   "temperature": TEMP, "top_p": TOP_P, "seed": sd, "max_new_tokens": 600}
            t0 = time.time()
            r = requests.post("http://127.0.0.1:8096/v1/tts", data=ormsgpack.packb(req), headers={"content-type": "application/msgpack"}, timeout=900)
            gen_s = time.time() - t0
            if not r.ok: print("ERR", lid, v, sd, r.status_code, r.text[:200], flush=True); continue
            open(p + ".tmp.wav", "wb").write(r.content); os.replace(p + ".tmp.wav", p)
            info = sf.info(p); dur = info.frames / info.samplerate
            rec = dict(variant="fish_fp8", tag=tag, voice=v, line=lid, cat=cat, seed=sd, frames=int(round(dur * 21.533)), dur=round(dur, 3),
                       gen_s=round(gen_s, 3), rtf=round(gen_s / max(dur, 1e-3), 3), stall=None, hit_max=dur >= 600 / 21.533 - 0.1,
                       peak_vram_gb=vram(), maxrss_gb=None, load_s=None, ram_cache=None, note="rtf includes ref encode + decode + HTTP")
            meta.write(json.dumps(rec) + "\n"); meta.flush(); print(rec, flush=True)
open(f"{ROOT}/PROGRESS.txt", "a").write(time.strftime("%F %T") + f" A2 gen {tag} complete\n")
