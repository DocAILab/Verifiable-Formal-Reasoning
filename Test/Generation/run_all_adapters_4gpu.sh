#!/usr/bin/env bash
set -euo pipefail

ROOT="${FVCODE_ROOT:-/home/zhangql24/FVCode}"
source "${ROOT}/.env.sh"

PYTHON_BIN="${FVCODE_PYTHON}"
MODEL="${MODEL:-/data/Qwen2.5-7B-Instruct}"
INPUT="${ROOT}/Data/ProverQA/test/golden/all.jsonl"
ADAPTER_ROOT="${ROOT}/Adapter"
OUTPUT_ROOT="${ROOT}/Output/Test/Generation/canonical_ab_681_ours0813"
LOG_ROOT="${ROOT}/Codex/eval_logs_canonical_ab_681_ours0813"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_one() {
  local gpu="$1" name="$2" adapter="$3"
  local output_dir="${OUTPUT_ROOT}/${name}"
  [[ -s "${adapter}/adapter_model.safetensors" ]] || {
    echo "ERROR missing adapter: ${adapter}" >&2
    return 2
  }
  echo "START gpu=${gpu} name=${name} date=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" Test/Generation/evaluate.py \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --input "${INPUT}" \
    --require_canonical_proof \
    --output_dir "${output_dir}" \
    --name "${name}" \
    --num_samples "${NUM_SAMPLES}" \
    --generation_batch_size 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.8 \
    --top_p 0.95 \
    >"${LOG_ROOT}/${name}.log" 2>&1
  echo "DONE gpu=${gpu} name=${name} date=$(date -Is)"
}

worker() {
  local gpu="$1"
  shift
  while [[ "$#" -gt 0 ]]; do
    local name="$1" adapter="$2"
    shift 2
    run_one "${gpu}" "${name}" "${adapter}"
  done
}

# Priority wave: step200, step600, step900, and step1000 start first on GPUs 0-3.
worker 0 \
  ours_step200  "${ADAPTER_ROOT}/Ours_0813/global_step_200" \
  ours_step100  "${ADAPTER_ROOT}/Ours_0813/global_step_100" \
  ours_step500  "${ADAPTER_ROOT}/Ours_0813/global_step_500" \
  >"${LOG_ROOT}/gpu0.driver.log" 2>&1 & p0=$!
worker 1 \
  ours_step600  "${ADAPTER_ROOT}/Ours_0813/global_step_600" \
  ours_step300  "${ADAPTER_ROOT}/Ours_0813/global_step_300" \
  ours_step700  "${ADAPTER_ROOT}/Ours_0813/global_step_700" \
  >"${LOG_ROOT}/gpu1.driver.log" 2>&1 & p1=$!
worker 2 \
  ours_step900  "${ADAPTER_ROOT}/Ours_0813/global_step_900" \
  ours_step400  "${ADAPTER_ROOT}/Ours_0813/global_step_400" \
  ours_step800  "${ADAPTER_ROOT}/Ours_0813/global_step_800" \
  >"${LOG_ROOT}/gpu2.driver.log" 2>&1 & p2=$!
worker 3 \
  ours_step1000 "${ADAPTER_ROOT}/Ours_0813/global_step_1000" \
  ours_step1098 "${ADAPTER_ROOT}/Ours_0813/global_step_1098" \
  warmup       "${ADAPTER_ROOT}/Warmup" \
  >"${LOG_ROOT}/gpu3.driver.log" 2>&1 & p3=$!

printf '%s\n' "${p0}" >"${LOG_ROOT}/gpu0.pid"
printf '%s\n' "${p1}" >"${LOG_ROOT}/gpu1.pid"
printf '%s\n' "${p2}" >"${LOG_ROOT}/gpu2.pid"
printf '%s\n' "${p3}" >"${LOG_ROOT}/gpu3.pid"

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]] || { echo "ERROR: one or more GPU workers failed" >&2; exit 1; }
echo "ALL_DONE date=$(date -Is)"
