#!/bin/bash
# fills transcripts into the server config, starts audiocpp_server, runs gen_gguf.py, stops it. Resumable.
cd ~/AI/drama3-community
python3 -c "
import json;c=open('scripts/gguf_server.tmpl.json').read()
for v in ('voiceA','voiceB','public'): c=c.replace('@@'+v, json.dumps(open(f'testset/{v}_ref.txt').read().strip())[1:-1])
open('runners/gguf_server.json','w').write(c)"
runners/audio.cpp/build/bin/audiocpp_server --config runners/gguf_server.json > logs/gguf_server.log 2>&1 &
SP=$!; for i in $(seq 120); do curl -s -o /dev/null http://127.0.0.1:8035/health && break; sleep 5; done
~/AI/tts/drama3/venv/bin/python scripts/gen_gguf.py gguf_q4k $1 >> logs/gen_gguf.log 2>&1
kill $SP
