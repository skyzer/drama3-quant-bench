"""A5 blind A/B/n page. usage: abpage.py SET VOICES... e.g. abpage.py public public ; abpage.py internal voiceA voiceB public
One take per (line, voice) per variant (seed 11), shuffled labels; key in ab/<SET>/key.json (open only after rating).
The rater scores each take 1-5 (naturalness + emotion fit) and can export ratings as JSON. Local file only, nothing is posted."""
import os, sys, json, random, subprocess, html
ROOT = os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0, ROOT + "/scripts")
from lines import LINES
SET, VOICES = sys.argv[1], sys.argv[2:]
VAR = [v for v in ["fish_fp8", "pack_fp8", "pack_int8_convrot", "pack_w6a8", "pack_w4a8", "int4_gptq", "gguf_q4k", "mlx_4bit"] if os.path.isdir(f"{ROOT}/out/{v}")]
D = f"{ROOT}/ab/{SET}"; os.makedirs(D + "/a", exist_ok=True); rnd = random.Random(20261015); key = {}; blocks = []
for lid, cat, text in LINES:
    for v in VOICES:
        takes = [(var, f"{ROOT}/out/{var}/{v}_{lid}_s11.wav") for var in VAR]; takes = [t for t in takes if os.path.exists(t[1])]
        if len(takes) < 2: continue
        rnd.shuffle(takes); iid = f"{v}_{lid}"; key[iid] = {}; cells = []
        for k, (var, p) in enumerate(takes):
            L = chr(65 + k); f = f"a/{iid}_{L}.mp3"; key[iid][L] = var
            if not os.path.exists(f"{D}/{f}"): subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-b:a", "128k", f"{D}/{f}"])
            cells.append(f'<div class=t><b>{L}</b><audio controls preload=none src="{f}"></audio>' + "".join(
                f'<label><input type=radio name="{iid}_{L}" value={s}>{s}</label>' for s in range(1, 6)) + "</div>")
        blocks.append(f'<section><h3>{iid} <small>[{cat}]</small></h3><p>{html.escape(text)}</p>{"".join(cells)}</section>')
json.dump(key, open(D + "/key.json", "w"), indent=1)
page = f"""<!doctype html><meta charset=utf-8><title>Drama 3 quant blind test ({SET})</title>
<style>body{{font:14px system-ui;max-width:1000px;margin:auto}}section{{border-bottom:1px solid #ddd;padding:8px}}.t{{display:flex;gap:8px;align-items:center}}audio{{height:28px}}</style>
<h1>Drama 3 quantization blind test: {SET}</h1><p>Each row's takes are shuffled, so the letters don't match model variants. Rate each take 1–5 for naturalness, emotion fit and voice match, then press Export. The key is in key.json. Don't open it until you've finished rating.</p>
{''.join(blocks)}<button onclick="let r={{}};document.querySelectorAll('input:checked').forEach(e=>r[e.name]=+e.value);
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r)]));a.download='ratings_{SET}.json';a.click()">Export ratings</button>
<script>document.querySelectorAll('input').forEach(e=>{{let k='{SET}:'+e.name;if(localStorage[k]==e.value)e.checked=true;e.onchange=()=>localStorage[k]=e.value}})</script>"""
open(D + "/index.html", "w").write(page); print(D, len(blocks), "items", VAR)
