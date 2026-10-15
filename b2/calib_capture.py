"""B2.1 capture MoE inputs on fp8 with emotion calibration prompts (voiceB voice + no-ref; never LJ test voice).
Saves b2/calib_acts.pt: per layer x[bf16 N,2048], idx[N,8], w[N,8], wt[N] (category weight)."""
import sys, os, random, time
ROOT = os.path.expanduser("~/AI/drama3-community"); COMFY = ROOT + "/comfy"
sys.path[:0] = [COMFY, ROOT + "/scripts", ROOT + "/pkg", ROOT + "/b2"]; sys.argv = [sys.argv[0]]
import torch, soundfile as sf
torch.set_grad_enabled(False)
import comfy.model_management as mm
from d3pack.drama3.loader import load_drama3, load_codec
from d3pack.drama3.generate import generate
from d3pack.drama3.prompt import build_prompt
from d3pack.drama3 import model as M
from d3pack.drama3.codec import SAMPLE_RATE
from calib_lines import CALIB
CAP = 8000; WT = {"neutral": 0.3}; rnd = random.Random(0)
store = {}; cur = {"wt": 1.0}; layer_of = {}
orig = M.SparseMoeBlock.forward
def fwd(self, x):
    flat = x.reshape(-1, x.shape[-1]); p = torch.softmax(self.gate(flat).float(), -1); tw, ti = p.topk(self.top_k, -1); tw = tw / tw.sum(-1, keepdim=True)
    L = layer_of[id(self)]; s = store.setdefault(L, {"x": [], "idx": [], "w": [], "wt": [], "n": 0})
    keep = [i for i in range(flat.shape[0]) if rnd.random() < 0.35]
    if keep and s["n"] < CAP:
        k = torch.tensor(keep, device=flat.device)
        s["x"].append(flat[k].to(torch.bfloat16).cpu()); s["idx"].append(ti[k].cpu()); s["w"].append(tw[k].cpu()); s["wt"].append(torch.full((len(keep),), cur["wt"])); s["n"] += len(keep)
    return orig(self, x)
M.SparseMoeBlock.forward = fwd
model = load_drama3(ROOT + "/models/drama3_fp8.safetensors", None, 0, experts_in_ram=False, expert_cache_gb=0.0)
codec = load_codec(ROOT + "/models/drama3_codec.safetensors"); mm.load_models_gpu(model.patchers())
blocks = [m for m in model.model.modules() if isinstance(m, M.SparseMoeBlock)] if hasattr(model, "model") else None
if not blocks:
    blocks = [m for p in model.patchers() for m in p.model.modules() if isinstance(m, M.SparseMoeBlock)]
for i, b in enumerate(blocks): layer_of[id(b)] = i
print("moe blocks", len(blocks), flush=True)
w, s = sf.read(ROOT + "/testset/voiceB_ref.wav", dtype="float32", always_2d=True); w = torch.from_numpy(w.mean(1))[None]
if s != SAMPLE_RATE:
    import torchaudio; w = torchaudio.functional.resample(w, s, SAMPLE_RATE)
mm.load_models_gpu([codec.patcher]); voiceB = (open(ROOT + "/testset/voiceB_ref.txt").read().strip(), codec.model.encode(w.to(codec.patcher.load_device))[0].cpu())
t0 = time.time()
for lid, cat, text in CALIB:
    for refs, sd in (([voiceB], 101), ([], 202)):
        cur["wt"] = WT.get(cat, 1.0); mm.load_models_gpu(model.patchers())
        fr = generate(model, build_prompt(model.tokenizer, model.config, text, refs, []), 400, 1.0, 30, 0.95, sd)
        print(lid, "ref" if refs else "noref", fr.shape[0], "frames", round(time.time() - t0), "s", "tokens L0", store.get(0, {}).get("n"), flush=True)
out = {L: {k: torch.cat(v[k]) for k in ("x", "idx", "w", "wt")} for L, v in store.items()}
torch.save(out, ROOT + "/b2/calib_acts.pt"); print("saved", {L: out[L]["x"].shape[0] for L in list(out)[:3]})
open(ROOT + "/PROGRESS.txt", "a").write(time.strftime("%F %T") + " B2.1 calib capture done\n")
