"""Mac MLX-audio Drama 3 generator. Resumable. usage: gen_mlx.py [TAG] [MODEL_DIR]
Runs on Apple Silicon only. Writes out/<tag>/... and meta.jsonl."""
import sys, os, json, time, resource
ROOT = os.path.expanduser("~/AI/drama3-community")  # mirrored; on MacBook use ~/AI/drama3-mlx/bench
# Allow override
ROOT = os.environ.get("D3_ROOT", os.path.expanduser("~/AI/drama3-mlx/bench"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# lines.py next to this script or in bench/scripts
for p in [os.path.dirname(__file__), os.path.join(ROOT,"scripts")]:
    if os.path.exists(os.path.join(p,"lines.py")): sys.path.insert(0,p); break
from lines import LINES, VOICES, SEEDS, TEMP, TOP_P
tag = sys.argv[1] if len(sys.argv)>1 else "mlx_4bit"
model_dir = sys.argv[2] if len(sys.argv)>2 else os.path.expanduser("~/AI/drama3-mlx/models/fish-audio-drama-3-4bit")
OUT = f"{ROOT}/out/{tag}"; os.makedirs(OUT, exist_ok=True)
from mlx_audio.tts.utils import load_model
import soundfile as sf, numpy as np
t0=time.time(); model=load_model(model_dir); load_s=time.time()-t0
refs={v:(f"{ROOT}/testset/{v}_ref.wav", open(f"{ROOT}/testset/{v}_ref.txt").read().strip()) for v in VOICES}
meta=open(OUT+"/meta.jsonl","a")
for lid,cat,text in LINES:
  for v in VOICES:
    for sd in SEEDS:
      p=f"{OUT}/{v}_{lid}_s{sd}.wav"
      if os.path.exists(p): continue
      w,t=refs[v]
      t1=time.time(); chunks=[]; 
      try:
        for result in model.generate(text=text, voice=None, ref_audio=w, ref_text=t, temperature=TEMP, top_p=TOP_P, top_k=30, seed=sd, max_tokens=600):
          if hasattr(result,"audio"): chunks.append(np.asarray(result.audio))
          elif hasattr(result,"samples"): chunks.append(np.asarray(result.samples))
      except TypeError:
        # older API kwargs
        for result in model.generate(text, ref_audio=w, ref_text=t, temperature=TEMP, top_p=TOP_P):
          chunks.append(np.asarray(getattr(result,"audio",getattr(result,"samples",result))))
      gen_s=time.time()-t1
      if not chunks: print("EMPTY",lid,v,sd,flush=True); continue
      wav=np.concatenate([c.reshape(-1) for c in chunks]); sr=getattr(model,"sample_rate",44100)
      sf.write(p+".tmp.wav",wav,sr); os.replace(p+".tmp.wav",p)
      dur=len(wav)/sr
      rec=dict(variant=tag,tag=tag,voice=v,line=lid,cat=cat,seed=sd,frames=int(round(dur*21.533)),dur=round(dur,3),
               gen_s=round(gen_s,3),rtf=round(gen_s/max(dur,1e-3),3),stall=None,hit_max=dur>=600/21.533-0.1,
               peak_vram_gb=None,maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,2),
               load_s=round(load_s,1),ram_cache=None,note="mlx-audio; maxrss is process RSS GB on macOS (bytes)")
      # macOS ru_maxrss is bytes
      if sys.platform=="darwin": rec["maxrss_gb"]=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,2)
      meta.write(json.dumps(rec)+"\n"); meta.flush(); print(rec,flush=True)
open(f"{ROOT}/PROGRESS.txt","a").write(time.strftime("%F %T")+f" A2 gen {tag} complete\n")
