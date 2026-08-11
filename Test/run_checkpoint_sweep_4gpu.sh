#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base model directory}"
ADAPTER_ROOT="${ADAPTER_ROOT:?Set ADAPTER_ROOT to the directory containing global_step_* adapters}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing dev.jsonl and test.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Output/Test/Generation}"
RUN_NAME="${RUN_NAME:-rule_grounded_process_rl}"
STEPS_TEXT="${STEPS:-100 200 300 400 500 600 700 800 900 1000 1098}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"

export PYTHONPATH="${REPO_ROOT}/Training:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

SHARD_ROOT="${OUTPUT_ROOT}/_shards/${RUN_NAME}"
LOG_ROOT="${OUTPUT_ROOT}/_logs/${RUN_NAME}"
mkdir -p "${SHARD_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

DATA_ROOT="${DATA_ROOT}" SHARD_ROOT="${SHARD_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
shard_root = Path(os.environ["SHARD_ROOT"])
for split in ("dev", "test"):
    rows = [
        line
        for line in (data_root / f"{split}.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    for shard in range(4):
        selected = rows[shard::4]
        (shard_root / f"{split}_shard{shard}.jsonl").write_text(
            "\n".join(selected) + ("\n" if selected else ""),
            encoding="utf-8",
        )
PY

run_shard() {
  local gpu="$1" step="$2" split="$3" shard="$4"
  local adapter="${ADAPTER_ROOT}/global_step_${step}"
  local slug="${RUN_NAME}_step${step}_${split}_shard${shard}"
  [[ -s "${adapter}/adapter_model.safetensors" ]] || {
    echo "Missing adapter: ${adapter}" >&2
    return 2
  }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${REPO_ROOT}/Test/Generation/evaluate.py" \
    --model "${MODEL_PATH}" \
    --adapter "${adapter}" \
    --input "${SHARD_ROOT}/${split}_shard${shard}.jsonl" \
    --output_dir "${OUTPUT_ROOT}/${slug}" \
    --name "${slug}" \
    --num_samples "${NUM_SAMPLES}" \
    --generation_batch_size 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.8 \
    --top_p 0.95 \
    >"${LOG_ROOT}/step${step}_${split}_shard${shard}.log" 2>&1
}

merge_split() {
  local step="$1" split="$2"
  local adapter="${ADAPTER_ROOT}/global_step_${step}"
  local slug="${RUN_NAME}_step${step}_${split}"
  local shard_dirs=()
  for shard in 0 1 2 3; do
    shard_dirs+=("${OUTPUT_ROOT}/${RUN_NAME}_step${step}_${split}_shard${shard}")
  done
  "${PYTHON_BIN}" "${REPO_ROOT}/Test/Generation/merge_generation_shards.py" \
    --output_dir "${OUTPUT_ROOT}/${slug}" \
    --name "${slug}" \
    --split "${split}" \
    --model "${MODEL_PATH}" \
    --adapter "${adapter}" \
    --input "${DATA_ROOT}/${split}.jsonl" \
    --num_samples "${NUM_SAMPLES}" \
    --shard_dirs "${shard_dirs[@]}" \
    >"${LOG_ROOT}/step${step}_${split}_merge.log" 2>&1
  "${PYTHON_BIN}" "${REPO_ROOT}/Test/Generation/summarize_uncertainty_subsets.py" \
    --output_dir "${OUTPUT_ROOT}/${slug}" \
    >"${LOG_ROOT}/step${step}_${split}_subsets.log" 2>&1
}

for step in ${STEPS_TEXT}; do
  echo "CHECKPOINT_START step=${step} date=$(date -Is)"
  pids=()
  for gpu in 0 1 2 3; do
    (run_shard "${gpu}" "${step}" dev "${gpu}"; run_shard "${gpu}" "${step}" test "${gpu}") &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ "${failed}" -eq 0 ]] || exit 1
  merge_split "${step}" dev
  merge_split "${step}" test
  echo "CHECKPOINT_DONE step=${step} date=$(date -Is)"
done

