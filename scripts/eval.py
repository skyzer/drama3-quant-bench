"""A3 per-take metrics, resumable (cache in eval/takes.jsonl keyed by tag/file). usage: eval.py TAG [TAG...]
Whisper large-v3 (faster-whisper, fp16, en, beam 5) WER with Whisper EnglishTextNormalizer; ECAPA (speechbrain voxceleb)
cosine to the reference clip; AST AudioSet tags (laughter/crying/screaming/whisper); RMS + pyin pitch stats."""
import sys, os, json, re, glob, numpy as np, torch, librosa, jiwer
ROOT = os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0, ROOT + "/scripts")
from lines import LINES
from whisper.normalizers import EnglishTextNormalizer
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier
from transformers import ASTFeatureExtractor, ASTForAudioClassification
norm = EnglishTextNormalizer(); TEXT = {l: t for l, _, t in LINES}; CAT = {l: c for l, c, _ in LINES}
strip = lambda t: re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]", " ", t)).strip()
dev = "cuda"
asr = WhisperModel("large-v3", device=dev, compute_type="float16")
spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": dev}, savedir=os.path.expanduser("~/.cache/sb_ecapa"))
fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
lab = ast.config.label2id
EV = {"laugh": ["Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Snicker"], "cry": ["Crying, sobbing", "Whimper"],
      "scream": ["Screaming", "Shout", "Yell"], "whisper": ["Whispering"]}
def load16(p): return librosa.load(p, sr=16000, mono=True)[0]
def emb(a): 
    with torch.no_grad(): return torch.nn.functional.normalize(spk.encode_batch(torch.tensor(a)[None].to(dev))[0, 0], dim=0)
REF = {v: emb(load16(f"{ROOT}/testset/{v}_ref.wav")) for v in ("voiceA", "voiceB", "public")}
def events(a):
    win = 16000 * 10; segs = [a[i:i + win] for i in range(0, max(1, len(a) - 8000), win)] or [a]
    with torch.no_grad():
        x = fe([s for s in segs], sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        pr = torch.sigmoid(ast(x).logits).max(0).values.cpu().numpy()
    return {k: float(max(pr[lab[n]] for n in ns if n in lab)) for k, ns in EV.items()}
def prosody(a):
    f = 480; fr = a[:len(a) // f * f].reshape(-1, f); r = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)
    act = r > r.max() - 40; v = r[act]
    f0, vf, _ = librosa.pyin(a, fmin=60, fmax=900, sr=16000, frame_length=1024, hop_length=320)
    f0 = f0[~np.isnan(f0)]; st = 12 * np.log2(f0 / 100) if len(f0) > 5 else np.array([np.nan])
    # trailing silence and longest internal silence (s)
    sil = ~act; tail = 0
    for s in sil[::-1]:
        if not s: break
        tail += 1
    run = mx = 0
    for s in sil: run = run + 1 if s else 0; mx = max(mx, run)
    return dict(rms_db=float(np.mean(v)), rms_p90=float(np.percentile(v, 90)), dyn_db=float(np.percentile(v, 90) - np.percentile(v, 10)),
                f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)) if len(st) > 5 else None, f0_std_st=float(np.nanstd(st)),
                f0_med_hz=float(np.median(f0)) if len(f0) else None, voiced=float(np.mean(~np.isnan(vf)) if vf is not None else 0),
                active=float(act.mean()), tail_sil_s=tail * 0.03, max_sil_s=mx * 0.03)
os.makedirs(ROOT + "/eval", exist_ok=True); cache = ROOT + "/eval/takes.jsonl"
done = set()
if os.path.exists(cache):
    for l in open(cache): done.add(json.loads(l)["key"])
out = open(cache, "a")
for tag in sys.argv[1:]:
    meta = {}
    for l in open(f"{ROOT}/out/{tag}/meta.jsonl"):
        m = json.loads(l); meta[f"{m['voice']}_{m['line']}_s{m['seed']}"] = m
    for p in sorted(glob.glob(f"{ROOT}/out/{tag}/*.wav")):
        name = os.path.basename(p)[:-4]; key = f"{tag}/{name}"
        if key in done or name.endswith(".tmp") or name not in meta: continue
        m = meta[name]; a = load16(p); dur = len(a) / 16000
        ref = strip(TEXT[m["line"]])
        if dur < 0.3: hyp = ""
        else:
            segs, _ = asr.transcribe(a, language="en", beam_size=5, condition_on_previous_text=False); hyp = " ".join(s.text for s in segs).strip()
        rn, hn = norm(ref), norm(hyp)
        if hn:
            msr = jiwer.process_words(rn, hn); N = max(1, msr.hits + msr.substitutions + msr.deletions)
            w = msr.wer; recall_w = (msr.substitutions + msr.deletions) / N; ins_r = msr.insertions / N
        else:
            w, recall_w, ins_r = 1.0, 1.0, 0.0
        sim = float(torch.dot(emb(a), REF[m["voice"]])) if dur >= 0.5 else None
        exp = len(ref.split()) * 0.33
        r = dict(key=key, **{k: m[k] for k in ("variant", "tag", "voice", "line", "cat", "seed", "dur", "gen_s", "rtf", "stall", "hit_max", "peak_vram_gb", "maxrss_gb")},
                 hyp=hyp, wer=round(w, 4), recall_wer=round(recall_w, 4), insertion_rate=round(ins_r, 4), spk_sim=sim, exp_dur=round(exp, 2), **(prosody(a) if dur >= 0.5 else {}), **(events(a) if dur >= 0.5 else {}))
        out.write(json.dumps(r) + "\n"); out.flush(); print(tag, name, r["wer"], r["spk_sim"] and round(r["spk_sim"], 3), hyp[:60], flush=True)
