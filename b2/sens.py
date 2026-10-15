"""B2.2 activation-aware, emotion-weighted sensitivity: per layer, relative MoE-output error of W4 vs W6 expert banks
on captured calibration tokens. Writes b2/sens.json. usage: sens.py [layers...]"""
import sys, os, json, time
ROOT = os.path.expanduser("~/AI/drama3-community"); TOOLS = ROOT + "/comfy/custom_nodes/ComfyUI-Drama3/tools"
sys.path[:0] = [ROOT + "/comfy", TOOLS, os.path.dirname(TOOLS)]; sys.argv = sys.argv[:1] + sys.argv[1:]
import torch, torch.nn.functional as F
torch.set_grad_enabled(False)
import convert as C
from comfy_kitchen.tensor.base import QuantizedTensor
SRC = os.environ.get("DRAMA3_HF_DIR", "checkpoints/drama-3"); dev = torch.device("cuda:0")
ck = C.Checkpoint(SRC); acts = torch.load(ROOT + "/b2/calib_acts.pt")
out_path = ROOT + "/b2/sens.json"; res = json.load(open(out_path)) if os.path.exists(out_path) else {}
layers = [int(a) for a in sys.argv[1:]] or range(40)
def deq(entries_params):
    _, params, qd = entries_params
    return [QuantizedTensor(q.to(dev), "AsymW4A8Int8Layout", p).dequantize().float() for q, p in zip(qd, params)]
for L in layers:
    if str(L) in res: continue
    t = time.time(); a = acts[L]; x = a["x"].to(dev).float(); idx = a["idx"].to(dev); w = a["w"].to(dev).float(); wt = a["wt"].to(dev)
    gu, dn = [], []
    for e in range(256):
        p = f"{C.LAYER_PREFIX}{L}.mlp.experts.{e}"
        f = {j: C.dequantize_fp8(ck.get(f"{p}.{j}.weight"), ck.get(f"{p}.{j}.weight_scale_inv"), dev) for j in ("gate_proj", "up_proj", "down_proj")}
        gu.append(torch.cat((f["gate_proj"], f["up_proj"]))); dn.append(f["down_proj"])
    def moe(GU, DN):
        y = torch.zeros_like(x)
        for e in range(256):
            m = (idx == e); rows = m.any(-1).nonzero().squeeze(-1)
            if rows.numel() == 0: continue
            g, u = (x[rows] @ GU[e].T).chunk(2, -1); h = (F.silu(g) * u) @ DN[e].T
            y[rows] += (w[rows] * m[rows]).sum(-1, keepdim=True) * h
        return y
    ref = moe(gu, dn); den = float((wt[:, None] * ref.pow(2)).sum()); r = {}
    emo = wt == 1.0
    for bits in (4, 6):
        GU = deq(C.w4a8_bank(gu, dev, bits)); DN = deq(C.w4a8_bank(dn, dev, bits))
        d = (moe(GU, DN) - ref).pow(2).sum(-1)
        r[f"err{bits}"] = float((wt * d).sum()) / den
        r[f"err{bits}_emo"] = float(d[emo].sum() / ref[emo].pow(2).sum())
        r[f"err{bits}_neu"] = float(d[~emo].sum() / ref[~emo].pow(2).sum()) if (~emo).any() else None
        del GU, DN
    r["n"] = int(x.shape[0]); res[str(L)] = r; json.dump(res, open(out_path, "w"), indent=1)
    print(L, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, round(time.time() - t), "s", flush=True)
    del gu, dn; torch.cuda.empty_cache()
open(ROOT + "/PROGRESS.txt", "a").write(time.strftime("%F %T") + " B2.2 sensitivity done\n")
