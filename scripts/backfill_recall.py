"""Backfill recall_wer + insertion_rate from stored hyp without re-running Whisper. Resumable overwrite of takes.jsonl."""
import json, os, re, sys
from whisper.normalizers import EnglishTextNormalizer
import jiwer
ROOT=os.path.expanduser("~/AI/drama3-community"); sys.path.insert(0,ROOT+"/scripts")
from lines import LINES
norm=EnglishTextNormalizer(); TEXT={l:t for l,_,t in LINES}
strip=lambda t: re.sub(r"\s+"," ",re.sub(r"\[[^\]]+\]"," ",t)).strip()
src=ROOT+"/eval/takes.jsonl"; tmp=src+".tmp"
n=0
with open(src) as f, open(tmp,"w") as o:
  for line in f:
    r=json.loads(line)
    if "recall_wer" not in r:
      ref=norm(strip(TEXT[r["line"]])); hyp=norm(r.get("hyp") or "")
      if hyp:
        m=jiwer.process_words(ref,hyp); N=max(1,m.hits+m.substitutions+m.deletions)
        r["recall_wer"]=round((m.substitutions+m.deletions)/N,4); r["insertion_rate"]=round(m.insertions/N,4)
      else:
        r["recall_wer"]=1.0; r["insertion_rate"]=0.0
      n+=1
    o.write(json.dumps(r)+"\n")
os.replace(tmp,src); print("backfilled",n)
