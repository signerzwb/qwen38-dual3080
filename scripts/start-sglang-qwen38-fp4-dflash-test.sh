#!/usr/bin/env bash
set -euo pipefail

CTX="${1:?usage: $0 <context-length> <max-total-tokens> [port]}"
MAX_TOTAL="${2:?usage: $0 <context-length> <max-total-tokens> [port]}"
PORT="${3:-8788}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-12}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
NO_VISION="${NO_VISION:-0}"

SRC=/home/dministrator/sglang-qwen38-fp4-test/sglang-src/python
PY=/home/dministrator/miniconda3/envs/sglang-awq-test/bin/python
MODEL=/home/dministrator/models/Qwen3.8-27B-Uncensored-NVFP4
DRAFT=/home/dministrator/models/Qwen3.8-27B-DFlash2
LOG=/home/dministrator/logs/sglang-qwen38-fp4-dflash-test.log

export PYTHONPATH="$SRC:/home/dministrator/flashinfer-0.6.17-test"
export PATH="/home/dministrator/miniconda3/envs/sglang-awq-test/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
export SGLANG_ENABLE_GRAPH_POOL_PRECARVE=1
export SGLANG_ENABLE_GRAPH_POOL_BORROW=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1
export SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1

mkdir -p /home/dministrator/logs

if [ "$NO_VISION" = "1" ]; then
  MM_FLAGS=(--language-only)
else
  MM_FLAGS=(--enable-multimodal)
fi

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3.8-27B-FP4-Dflash-test \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tp-size 2 \
  --trust-remote-code \
  ${MM_FLAGS[@]} \
  --context-length "$CTX" \
  --max-total-tokens "$MAX_TOTAL" \
  --max-running-requests 1 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --kv-cache-dtype fp8_e4m3 \
  --enable-mixed-chunk \
  --chunked-prefill-size 2048 \
  --disable-prefill-cuda-graph \
  --attention-backend flashinfer \
  --mamba-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 8 \
  --mamba-max-states-per-path 6 \
  --mamba-track-interval 512 \
  --page-size 64 \
  --radix-eviction-policy lru \
  --enable-hierarchical-cache \
  --hicache-size "$HICACHE_SIZE_GB" \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_back \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-window-size 2048 \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --enable-linear-replayssm-spec \
  > "$LOG" 2>&1
