import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
R=os.path.expanduser("~/AI/drama3-community"); d=pd.read_csv(R+"/results_by_cat.csv")
d=d[d.tag.isin(["fish_fp8","pack_fp8","pack_int8_convrot","pack_w6a8","pack_w4a8","gguf_q4k","b2_emo_a_full"])]
for m in ["fail","target_ev","dyn_db","f0_range_st","wer_c"]:
    p=d.pivot(index="cat",columns="tag",values=m); p.plot.bar(figsize=(10,4)); plt.title(m+" by category (seed-matched, 630 takes/variant)"); plt.tight_layout(); plt.savefig(f"{R}/charts/cat_{m}.png",dpi=120); plt.close()
print("ok")
