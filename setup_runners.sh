#!/bin/bash
# A1c: int4 runner (sglang-omni) and GGUF runner (audio.cpp drama-3-native @4dbb4d1). niced to protect gen timing. Resumable.
cd ~/AI/drama3-community; mkdir -p runners; cd runners; U=~/.local/bin/uv
[ -d sglang-omni ] || git clone -q https://github.com/sgl-project/sglang-omni.git
(cd sglang-omni && git log -1 --format="sglang-omni %h %ci" && { [ -x .venv/bin/python ] || $U venv -q -p 3.12 .venv; } && nice -n 19 $U pip install -q -p .venv/bin/python --prerelease=allow -e ".[fishaudio]" && echo "$(date "+%F %T") A1c sglang-omni installed" >> ../../PROGRESS.txt)
[ -d audio.cpp ] || git clone -q -b drama-3-native https://github.com/JarodMica/audio.cpp.git
cd audio.cpp && git checkout -q 4dbb4d128f8f5ae373067f7213cbe340258af413; git log -1 --format="audio.cpp %h %ci"
ls; sed -n 1,80p docs/models/fish_drama3.md | grep -n -i -E "cmake|build|linux" | head
../sglang-omni/.venv/bin/pip --version >/dev/null 2>&1
$U pip install -q -p ../sglang-omni/.venv/bin/python cmake ninja
export PATH=$PWD/../sglang-omni/.venv/bin:$PATH
nice -n 19 cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && nice -n 19 cmake --build build -j 8 && echo "$(date "+%F %T") A1c audio.cpp built" >> ../../PROGRESS.txt || echo "$(date "+%F %T") A1c audio.cpp build FAILED (see logs/runners.log)" >> ../../PROGRESS.txt
