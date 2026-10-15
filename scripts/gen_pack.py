"""A2 generator for ComfyUI-Drama3 files. Resumable: skips takes whose wav exists.
usage: gen_pack.py VARIANT MODELFILE [--ram CACHE_GB] [--subset N] [--tag NAME]
Writes out/<tag>/<voice>_<line>_s<seed>.wav and out/<tag>/meta.jsonl"""
import sys, os, json, time, logging, argparse
ROOT = os.path.expanduser("~/AI/drama3-community"); COMFY = ROOT + "/comfy"
ap = argparse.ArgumentParser(); ap.add_argument("variant"); ap.add_argument("model")
ap.add_argument("--ram", type=float, default=None); ap.add_argument("--subset", type=int, default=0); ap.add_argument("--tag")
a = ap.parse_args(); sys.argv = [sys.argv[0]]
sys.path[:0] = [COMFY, ROOT + "/scripts", ROOT + "/pkg"]
import torch, numpy as np, soundfile as sf, resource
torch.set_grad_enabled(False)
import comfy.model_management as mm
from d3pack.drama3.loader import load_drama3, load_codec
from d3pack.drama3.generate import generate
from d3pack.drama3.prompt import build_prompt
from d3pack.drama3.codec import SAMPLE_RATE
from lines import LINES, VOICES, SEEDS, TEMP, TOP_P
if os.environ.get("D3_CATS"): LINES = [l for l in LINES if l[1] in os.environ["D3_CATS"].split(",")]
if os.environ.get("D3_SEEDS"): SEEDS = [int(s) for s in os.environ["D3_SEEDS"].split(",")]
TOP_K = 30
stalls = []
class H(logging.Handler):
    def emit(self, r):
        if "went silent" in r.getMessage(): stalls.append(1)
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.INFO)
tag = a.tag or a.variant; OUT = f"{ROOT}/out/{tag}"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
model = load_drama3(a.model, None, 0, experts_in_ram=a.ram is not None, expert_cache_gb=a.ram or 0.0)
codec = load_codec(ROOT + "/models/drama3_codec.safetensors")
mm.load_models_gpu(model.patchers()); load_s = time.time() - t0
dev = codec.patcher.load_device
def enc(v):
    w, s = sf.read(f"{ROOT}/testset/{v}_ref.wav", dtype="float32", always_2d=True); w = torch.from_numpy(w.mean(1))[None]
    if s != SAMPLE_RATE:
        import torchaudio; w = torchaudio.functional.resample(w, s, SAMPLE_RATE)
    mm.load_models_gpu([codec.patcher])
    return (open(f"{ROOT}/testset/{v}_ref.txt").read().strip(), codec.model.encode(w.to(dev))[0].cpu())
refs = {v: enc(v) for v in VOICES}
lines = LINES[::max(1, len(LINES)//a.subset)][:a.subset] if a.subset else LINES
seeds = SEEDS[:1] if a.subset else SEEDS
meta = open(OUT + "/meta.jsonl", "a"); torch.cuda.reset_peak_memory_stats()
for lid, cat, text in lines:
    for v in VOICES:
        for sd in seeds:
            p = f"{OUT}/{v}_{lid}_s{sd}.wav"
            if os.path.exists(p): continue
            stalls.clear(); prompt = build_prompt(model.tokenizer, model.config, text, [refs[v]], [])
            mm.load_models_gpu(model.patchers()); torch.cuda.synchronize(); t = time.time()
            fr = generate(model, prompt, 600, TEMP, TOP_K, TOP_P, sd)
            torch.cuda.synchronize(); gen_s = time.time() - t
            codes = fr[:, 1:].T.contiguous()
            mm.load_models_gpu([codec.patcher])
            wav = codec.model.decode(codes[None].to(dev), torch.float32).float().cpu().reshape(-1).numpy() if codes.shape[-1] else np.zeros(2048, np.float32)
            sf.write(p + ".tmp.wav", wav, SAMPLE_RATE); os.replace(p + ".tmp.wav", p)
            dur = len(wav) / SAMPLE_RATE
            rec = dict(variant=a.variant, tag=tag, voice=v, line=lid, cat=cat, seed=sd, frames=int(codes.shape[-1]), dur=round(dur, 3),
                       gen_s=round(gen_s, 3), rtf=round(gen_s / max(dur, 1e-3), 3), stall=bool(stalls), hit_max=codes.shape[-1] >= 600,
                       peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                       load_s=round(load_s, 1), ram_cache=a.ram)
            meta.write(json.dumps(rec) + "\n"); meta.flush(); print(rec, flush=True)
open(f"{ROOT}/PROGRESS.txt", "a").write(time.strftime("%F %T") + f" A2 gen {tag} complete\n")
