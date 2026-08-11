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
