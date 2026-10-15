#!/bin/bash
cd ~/AI/drama3-community
score_one() {
  local tag=$1
  evalenv/bin/python scripts/backfill_recall.py >/dev/null
  evalenv/bin/python scripts/listen_queue.py pack_fp8 pack_int8_convrot pack_w6a8 pack_w4a8 fish_fp8 gguf_q4k >/dev/null 2>&1
  evalenv/bin/python scripts/analyze.py > logs/analyze_$tag.txt 2>&1
  echo "$(date "+%F %T") A4 scored $tag" >> PROGRESS.txt
}
for tag in pack_w4a8 fish_fp8 gguf_q4k; do
  # skip if already scored
  grep -q "A4 scored $tag" PROGRESS.txt && continue
  echo "$(date "+%F %T") A4 waiting for $tag" >> PROGRESS.txt
  while true; do
    n=$(wc -l < out/$tag/meta.jsonl 2>/dev/null || echo 0)
    e=$(python3 -c "import json;print(sum(1 for l in open(\"eval/takes.jsonl\") if json.loads(l).get(\"tag\")==\"$tag\"))" 2>/dev/null || echo 0)
    echo "$(date +%T) $tag gen=$n eval=$e"
    if [ "$n" -ge 630 ] && [ "$e" -ge 630 ]; then score_one $tag; break; fi
    sleep 60
  done
done
echo "$(date "+%F %T") A4 watch_score done" >> PROGRESS.txt
