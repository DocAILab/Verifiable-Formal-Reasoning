# Evaluation

The evaluator performs one-shot full structured generation with the same
prompt, parser, Z3 verifier, rule ontology, and RuleChecker used by training.

It reports answer `Avg@3`, `Pass@3`, format validity, semantic and rule-aware
verification rates without cascade, rule-grounded step rates, and Reasoning
Granularity Deviation (RGD). `summarize_uncertainty_subsets.py` writes paired
summaries with and without Uncertain-label problems.

```bash
export PYTHONPATH="$PWD/Training:$PWD"
python Test/Generation/evaluate.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --adapter /path/to/adapter \
  --input /path/to/test.jsonl \
  --output_dir Output/Test/example \
  --num_samples 3 \
  --max_new_tokens 4096

python Test/Generation/summarize_uncertainty_subsets.py \
  --output_dir Output/Test/example
```

Run the CPU reward and closure tests with:

```bash
pytest -q Test/tests
```

For a four-GPU checkpoint sweep, set `MODEL_PATH`, `ADAPTER_ROOT`, and
`DATA_ROOT`, then run `bash Test/run_checkpoint_sweep_4gpu.sh`. The defaults
match the experiment protocol: three samples, 4,096 generated tokens,
temperature 0.8, and top-p 0.95.

## FOLIO

Use the prepared validation subset, not `converted_validation.jsonl`. The raw
conversion puts the query in `nl2fol` and packs all premises into one entry.
The evaluator rejects that input before loading a model. Raw files are preserved.

```bash
python Test/Generation/prepare_folio.py
python Test/Generation/evaluate.py \
  --dataset folio \
  --input Data/FOLIO/verified_ab_v1/validation.jsonl \
  --validate_only

python Test/Generation/evaluate.py \
  --dataset folio \
  --model /path/to/Qwen3-8B \
  --adapter /path/to/adapter \
  --thinking_mode native \
  --input Data/FOLIO/verified_ab_v1/validation.jsonl \
  --output_dir Output/Test/FOLIO/qwen3_adapter \
  --num_samples 3 --generation_batch_size 1 \
  --max_new_tokens 4096 --temperature 0.8 --top_p 0.95
```

Omit `--adapter` for a base model. Use `--thinking_mode explicit` for the
Qwen2.5 non-native-thinking experiment. The shared training prompt, ontology,
Z3 verifier and RuleChecker are unchanged. Pass the generation limit explicitly:
the legacy evaluator default is still 512, whereas these experiments use 4096.
Use a separate output directory per dataset/model/checkpoint/settings combination.

`--dataset auto` recognizes prepared FOLIO. Canonical proofs remain required by
default for ProverQA, but are not required for FOLIO. Explicitly requiring them
on FOLIO fails instead of fabricating proofs. `--validate_only` needs neither
model weights nor GPU/Transformers; it checks the prepared-data schema and
normalization. Z3 consistency/label checks are performed during preparation.

FOLIO reports Avg@3, Pass@3, format, Z3 verification without cascade, and
rule-grounded metrics under the existing closed rule ontology. RGD is JSON
`null` / Markdown `N/A` without a reference proof. Mixed-dataset summaries
calculate RGD only over responses with references and report coverage.
Unannotated difficulty is `unknown`; no Easy/Medium/Hard categories are invented.
This is a curated supported A/B subset, **not a full official FOLIO score**.
Existential/equality reasoning is not fully covered by the training rule ontology;
Z3 compatibility does not imply complete RuleChecker coverage.

The preparation manifest and per-row exclusion reasons are described in
[`Data/FOLIO/README.md`](../Data/FOLIO/README.md). Train and validation remain
separate; use validation for cross-dataset testing, not a merged train/test set.

The existing four-GPU sweep also accepts FOLIO without a second launcher:

```bash
DATASET=folio \
EVAL_SPLITS=validation \
THINKING_MODE=native \
DATA_ROOT="$PWD/Data/FOLIO/verified_ab_v1" \
MODEL_PATH=/path/to/Qwen3-8B \
ADAPTER_ROOT=/path/to/adapters \
OUTPUT_ROOT="$PWD/Output/Test/FOLIO" \
RUN_NAME=qwen3_folio \
STEPS="100 200 300" \
bash Test/run_checkpoint_sweep_4gpu.sh
```

The sweep evaluates each checkpoint with four input shards and merges all
samples. `DATASET=folio` defaults to `validation`; other datasets keep the
legacy `dev test` default, overridable with `EVAL_SPLITS`. This script expects
`ADAPTER_ROOT/global_step_N/adapter_model.safetensors` as before.
