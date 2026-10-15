"""A4 aggregate. Seed-matched complete items only. stall_cut vs energy_silent. Includes recall_wer."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.expanduser("~/AI/drama3-community"); BASE = "pack_fp8"
SEEDS = {11,22,33,44,55,66,77}
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
else: d["recall_wer"] = d.recall_wer.clip(upper=1.0)
for c in ["spk_sim","rms_db","dyn_db","f0_range_st","f0_std_st","rtf","insertion_rate"]:
    if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
# seed-matched items present in ALL finished tags (630 takes)
finished = [t for t,n in d.groupby("tag").size().items() if n >= 630]
keep = None
for t in finished:
    g = d[d.tag == t]
    ok = {(ln,v) for (ln,v),gg in g.groupby(["line","voice"]) if set(gg.seed) >= SEEDS}
    keep = ok if keep is None else (keep & ok)
if keep is None: keep = set()
# only include a tag if it has all SEEDS for every kept item
rows_ok=[]
for tname,g in d.groupby("tag"):
  if keep and all(set(g[(g.line==ln)&(g.voice==v)].seed)>=SEEDS for ln,v in keep):
    rows_ok.append(g[g.apply(lambda r: (r.line,r.voice) in keep,1)])
d_m = pd.concat(rows_ok) if rows_ok else d.iloc[:0].copy()
M = ["wer_c","recall_wer","fail","energy_silent","stall_cut","truncated","babble","spk_sim","rms_db","dyn_db","f0_range_st","target_ev","rtf"]
M = [m for m in M if m in d_m.columns]
rng = np.random.default_rng(0)
def boot(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    b = rng.choice(x, (n, len(x))).mean(1); return tuple(np.percentile(b, [2.5, 97.5]))
items = d_m.groupby(["tag","line","voice","cat"], as_index=False)[M].mean(numeric_only=True) if len(d_m) else pd.DataFrame()
rows = []
for tag, g in items.groupby("tag"):
    r = dict(variant=tag, n_items=len(g), n_takes=int(((d.tag==tag)&d.apply(lambda x:(x.line,x.voice) in keep,1)).sum()))
    for m in M:
        r[m] = g[m].mean(); lo,hi = boot(g[m]); r[m+"_lo"], r[m+"_hi"] = lo, hi
    rows.append(r)
res = pd.DataFrame(rows)
if BASE in set(res.variant) and len(items):
    b = items[items.tag==BASE].set_index(["line","voice"])
    for i, r in res.iterrows():
        g = items[items.tag==r.variant].set_index(["line","voice"]); j = g.index.intersection(b.index)
        for m in [x for x in ["wer_c","recall_wer","fail","spk_sim","dyn_db","f0_range_st","target_ev","energy_silent","stall_cut"] if x in M]:
            diff = (g.loc[j,m] - b.loc[j,m]).values; res.loc[i,"d_"+m] = np.nanmean(diff) if len(j) else np.nan
            lo,hi = boot(diff); res.loc[i,"d_"+m+"_lo"], res.loc[i,"d_"+m+"_hi"] = lo, hi
sizes = {"pack_fp8":"drama3_fp8","pack_int8_convrot":"drama3_int8_convrot","pack_w6a8":"drama3_w6a8","pack_w4a8":"drama3_w4a8"}
for i,r in res.iterrows():
    f = f"{ROOT}/models/{sizes.get(r.variant,'')}.safetensors"
    res.loc[i,"file_gb"] = round(os.path.getsize(f)/1e9,2) if os.path.exists(f) else np.nan
    t = d[d.tag==r.variant]; res.loc[i,"peak_vram_gb"]=t.peak_vram_gb.max(); res.loc[i,"maxrss_gb"]=t.maxrss_gb.max()
res.round(4).to_csv(ROOT+"/results.csv", index=False)
if len(items):
    items.groupby(["tag","cat"])[M].mean(numeric_only=True).reset_index().round(4).to_csv(ROOT+"/results_by_cat.csv", index=False)
    items.groupby(["tag","voice"])[M].mean(numeric_only=True).reset_index().round(4).to_csv(ROOT+"/results_by_voice.csv", index=False)
d.to_csv(ROOT+"/eval/takes_flagged.csv", index=False)
os.makedirs(ROOT+"/charts", exist_ok=True)
order = [t for t in ["fish_fp8","pack_fp8","pack_int8_convrot","pack_w6a8","pack_w4a8","gguf_q4k","mlx_4bit","sgl_int4"] if t in set(res.variant)]
if order:
    R = res.set_index("variant").loc[order]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a,(m,t) in zip(ax,[("wer_c","WER"),("recall_wer","recall WER (S+D)/N"),("fail","fail rate"),("spk_sim","speaker sim")]):
        if m not in R: continue
        a.bar(order, R[m], yerr=[R[m]-R[m+"_lo"], R[m+"_hi"]-R[m]], capsize=3, color="#e8912d"); a.set_title(t); a.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(ROOT+"/charts/quality.png", dpi=120); plt.close()
print(res.to_string())
print("finished tags", finished, "matched items", len(keep))
