#!/usr/bin/env bash
# pass@1 evaluation via eval_pass1.py (greedy decoding).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT:?CKPT is required}"
: "${DATA:?DATA is required}"

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/results/$(basename "${CKPT}")}
N_GPUS=${N_GPUS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

cd "${PROJECT_ROOT}"
exec python3 src/eval_pass1.py \
  --ckpt "${CKPT}" \
  --data "${DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --tensor-parallel-size "${N_GPUS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
