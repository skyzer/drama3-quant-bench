"""Build a listen queue of sob/soblaugh failures (and matched ok controls) for the public voice.
Writes eval/listen_queue.csv + copies wavs to eval/listen/<id>.wav. Human labels go in eval/listen_labels.json.
usage: listen_queue.py [TAG...]  (default: all scored tags)"""
import os,sys,json,glob,shutil,csv,random
ROOT=os.path.expanduser("~/AI/drama3-community")
tags=sys.argv[1:] or [os.path.basename(p) for p in glob.glob(ROOT+"/out/*") if os.path.isdir(p) and not os.path.basename(p).startswith(("smoke","prof_"))]
takes=[json.loads(l) for l in open(ROOT+"/eval/takes.jsonl") if json.loads(l)["tag"] in tags]
# recompute flags loosely matching analyze
rows=[]
for r in takes:
  if r.get("voice")!="public" or r.get("cat") not in ("sob","soblaugh"): continue
  stall=bool(r.get("stall")); active=r.get("active") or 0; sil=max(r.get("max_sil_s") or 0,r.get("tail_sil_s") or 0)
  energy_silent=(active<0.45) or (sil>2.0) or ((r.get("dur") or 0)<0.5)
  fail=energy_silent or stall or (r.get("wer") or 0)>=0.4 or (r.get("hit_max") is True)
  r2=dict(r,energy_silent=energy_silent,stall_cut=stall,fail=fail); rows.append(r2)
rnd=random.Random(20261015)
fails=[r for r in rows if r["fail"]]; oks=[r for r in rows if not r["fail"]]
# up to 30 fails + 10 ok controls, stratified by tag then cat
pick=[]; 
for tag in sorted({r["tag"] for r in fails}):
  for cat in ("sob","soblaugh"):
    bucket=[r for r in fails if r["tag"]==tag and r["cat"]==cat]
    rnd.shuffle(bucket); pick.extend(bucket[:max(1, 30//max(1,len({x["tag"] for x in fails})*2))])
pick=pick[:30]
controls=[]; 
for tag in sorted({r["tag"] for r in oks}):
  bucket=[r for r in oks if r["tag"]==tag]; rnd.shuffle(bucket); controls.extend(bucket[:2])
controls=controls[:10]
# shuffle for blind listen; key separate
items=[{"role":"fail",**r} for r in pick]+[{"role":"ok",**r} for r in controls]
rnd.shuffle(items)
D=ROOT+"/eval/listen"; os.makedirs(D,exist_ok=True)
key=[]; qrows=[]
for i,r in enumerate(items,1):
  src=f"{ROOT}/out/{r['tag']}/{r['voice']}_{r['line']}_s{r['seed']}.wav"
  dst=f"{D}/{i:03d}.wav"
  if os.path.exists(src):
    if not os.path.exists(dst): shutil.copy2(src,dst)
    key.append({"id":f"{i:03d}","role":r["role"],"tag":r["tag"],"line":r["line"],"cat":r["cat"],"seed":r["seed"],
                "wer":r.get("wer"),"stall_cut":r["stall_cut"],"energy_silent":r["energy_silent"],"hyp":r.get("hyp","")[:120]})
    qrows.append({"id":f"{i:03d}","wav":f"{i:03d}.wav","cat":r["cat"],"line_id":r["line"],
                  "prompt_hint":"(open lines.py for full text; do not peek at key.json)",
                  "label_intelligible":"","label_emotion_present":"","label_emotion_natural":"","label_notes":""})
json.dump(key,open(D+"/key.json","w"),indent=1)
with open(ROOT+"/eval/listen_queue.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(qrows[0])); w.writeheader(); w.writerows(qrows)
# static listen page
open(D+"/index.html","w").write("""<!doctype html><meta charset=utf-8><title>Sob listen pass (public voice)</title>
<style>body{font:14px system-ui;max-width:900px;margin:auto}audio{width:100%}section{border-bottom:1px solid #ddd;padding:8px}</style>
<h1>Listen pass: sob / soblaugh failures + controls (public voice only)</h1>
<p>Rate each clip. Do not open key.json until finished. Export fills localStorage.</p>
""" + "".join(f'<section><h3>{r["id"]} [{r["cat"]}] {r["line_id"]}</h3><audio controls src="{r["wav"]}"></audio>'
  f'<div>intelligible? <select id="{r["id"]}_int"><option></option><option>yes</option><option>partial</option><option>no</option></select> '
  f'emotion present? <select id="{r["id"]}_ep"><option></option><option>yes</option><option>weak</option><option>no</option></select> '
  f'natural? <select id="{r["id"]}_en"><option></option><option>yes</option><option>forced</option><option>broken</option></select></div></section>'
  for r in qrows) + """
<button onclick="let o={};document.querySelectorAll('select').forEach(e=>o[e.id]=e.value);
localStorage['listen_labels']=JSON.stringify(o);let a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)]));a.download='listen_labels.json';a.click()">Export labels</button>
<script>let s=JSON.parse(localStorage.listen_labels||'{}');Object.entries(s).forEach(([k,v])=>{let e=document.getElementById(k);if(e)e.value=v})</script>
""")
print("listen queue",len(qrows),"->",D,"fails",len(pick),"ok",len(controls))
