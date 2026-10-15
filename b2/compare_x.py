"""A4 aggregate. Seed-matched complete items only. stall_cut vs energy_silent. Includes recall_wer."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.expanduser("~/AI/drama3-community"); BASE = "pack_fp8"
SEEDS = set(range(11,78,11)) | set(range(88,155,11))
d = pd.DataFrame([json.loads(l) for l in open(ROOT + "/eval/takes.jsonl")])
d = d[~d.tag.str.startswith(("smoke", "prof_"))].copy()
nw = lambda s: len(str(s).split())
d["ref_words"] = d.exp_dur / 0.33
d["stall_cut"] = d.stall.fillna(False).astype(bool).astype(float)
d["energy_silent"] = ((d.active.fillna(0) < 0.45) | (d.max_sil_s.fillna(0) > 2.0) | (d.tail_sil_s.fillna(0) > 2.0) | (d.dur < 0.5)).astype(float)
d["truncated"] = ((d.energy_silent == 0) & (d.wer >= 0.4) & (d.hyp.map(nw) < 0.6 * d.ref_words)).astype(float)
d["babble"] = ((d.energy_silent == 0) & (d.truncated == 0) & ((d.hit_max == True) | (d.dur > 3 * d.exp_dur + 3) | (d.wer >= 0.4))).astype(float)
d["fail"] = ((d.energy_silent + d.truncated + d.babble + d.stall_cut) > 0).astype(float)
TGT = {"scream": "scream", "sob": "cry", "laugh": "laugh", "whisper": "whisper"}
def target(r):
    if r["cat"] == "soblaugh": return max(r.get("laugh", 0) or 0, r.get("cry", 0) or 0)
    return r.get(TGT[r["cat"]], np.nan) if r["cat"] in TGT else np.nan
d["target_ev"] = d.apply(target, axis=1)
d["wer_c"] = d.wer.clip(upper=1.0)
if "recall_wer" not in d: d["recall_wer"] = d.wer_c  # fallback until re-eval
import sys
d["tag"] = d.tag.str.replace("^xsob_", "", regex=True)
CATS = sys.argv[1].split(","); TAGS = sys.argv[2:]
s = d[d.cat.isin(CATS) & d.tag.isin(TAGS)]
items = None
for t in TAGS:
    g = s[s.tag == t]; ok = {(ln, v) for (ln, v), gg in g.groupby(["line", "voice"]) if set(gg.seed) >= SEEDS}
    items = ok if items is None else items & ok
s = s[s.apply(lambda r: (r.line, r.voice) in items, 1)]
rng = np.random.default_rng(0); it = sorted(items)
print("cats", CATS, "items", len(it), "takes/tag", len(s) // len(TAGS))
per = {t: s[s.tag == t].groupby(["line", "voice"])[["fail", "stall_cut", "energy_silent", "wer_c", "target_ev", "spk_sim"]].mean() for t in TAGS}
for t in TAGS:
    m = per[t].mean(); print(f"{t:22s} " + " ".join(f"{k}={m[k]:.3f}" for k in m.index))
for base in ("pack_fp8", "pack_w4a8"):
    if base not in TAGS: continue
    for t in TAGS:
        if t in ("pack_fp8", "pack_w4a8"): continue
        dd = (per[t]["fail"] - per[base]["fail"]).values
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        print(f"  d_fail {t} - {base}: {dd.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
