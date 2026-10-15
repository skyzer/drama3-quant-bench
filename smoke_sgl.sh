#!/bin/bash
C=$HOME/AI/drama3-community/runners/sglang-omni/.venv/lib/python3.12/site-packages/nvidia/cu13
V=$HOME/AI/drama3-community/runners/sglang-omni/.venv/bin
export CUDA_HOME=$C PATH=$V:$C/bin:$PATH CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13
export CPATH=$C/include CPLUS_INCLUDE_PATH=$C/include LIBRARY_PATH=$C/lib
rm -rf ~/.cache/sglang/.cache/flashinfer
cd ~/AI/drama3-community/runners/sglang-omni
.venv/bin/sgl-omni serve --model-path ../../models/drama-3-int4 --port 8011 --tts_engine.gpu_memory_fraction 0.3 --vocoder.gpu_memory_fraction 0.1
