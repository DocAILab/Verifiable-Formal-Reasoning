# Canonical formal-proof test subset

These files augment `Data/ProverQA/test/all.jsonl` without modifying the source test set.

- `all.jsonl`: 681 examples with a locally verified `canonical_proofs` annotation.
- `easy.jsonl`, `medium.jsonl`, `hard.jsonl`: the same accepted examples split by difficulty.
- `excluded.jsonl`: 129 source examples that are not part of the canonical-proof subset.
- `manifest.json`: coverage, difficulty counts, proof-length statistics, and validation metadata.

Every accepted proof was proposed by `deepseek-v4-pro` and then independently accepted by the project schema checks, Z3 verifier, RuleChecker, nontrivial-progress checks, answer binding, and final dependency-closure checks. Model output alone is never treated as a golden proof.

`proof_length` counts non-`GOAL_BINDING` generated actions in the final dependency closure. Premises and the terminal `GOAL_BINDING` action are excluded. Evaluation uses the same closure-based action count for generated responses.

All reported metrics, including answer accuracy, pass@3, format rate, verifier rates, rule-grounded rates, and granularity error, must use the same 681-example `all.jsonl` subset. Do not combine 810-example accuracy metrics with 681-example granularity metrics.

Current coverage is 681/810 (84.07%): Easy 261/280, Medium 221/249, and Hard 199/281. Of the 129 exclusions, 12 have a gold target that is not entailed by the supplied `nl2fol` premises; 117 exhausted proof generation without a fully RuleChecker-valid trajectory.

The generation and CPU smoke scripts live in `Codex/build_golden_formal_proofs.py` and `Codex/test_golden_formal_proofs_smoke.py`. Intermediate candidates and verifier logs stay under ignored `Codex/golden_formal_proofs*/` directories.
