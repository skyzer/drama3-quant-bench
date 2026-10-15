"""B2.3 build a mixed-precision expert file: W6A8 on the chosen layers, W4A8 elsewhere; dense bf16 or int8.
usage: build_mixed.py OUT --w6 0,3,7 [--dense int8]"""
import sys, os, argparse, json
ROOT = os.path.expanduser("~/AI/drama3-community"); TOOLS = ROOT + "/comfy/custom_nodes/ComfyUI-Drama3/tools"
sys.path[:0] = [ROOT + "/comfy", TOOLS, os.path.dirname(TOOLS)]
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--w6", default=""); ap.add_argument("--dense", default="bf16")
a = ap.parse_args(); sys.argv = sys.argv[:1]
import torch; torch.set_grad_enabled(False)
import convert as C
W6 = {int(i) for i in a.w6.split(",") if i}; calls = {"n": 0}; orig = C.w4a8_bank
def bank(weights, device, bits=4):
    L = calls["n"] // 2; calls["n"] += 1
    return orig(weights, device, 6 if L in W6 else 4)
C.w4a8_bank = bank
ow = C.write
def write(t, path, md):
    md["drama3_mixed_w6_layers"] = json.dumps(sorted(W6)); md["drama3_recipe"] = "emotion-calibrated mixed W4A8/W6A8 (Skysay B2)"; ow(t, path, md)
C.write = write
C.convert_model(argparse.Namespace(src=os.environ.get("DRAMA3_HF_DIR", "checkpoints/drama-3"), out=a.out, format="w4a8", dense=a.dense, device="cuda:0"))
