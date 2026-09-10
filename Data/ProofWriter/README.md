# ProofWriter OWA Binary Evaluation Data

## Scope

`owa_binary_v1` is a cross-dataset evaluation set built from the official
ProofWriter V2020.12.3 OWA **test** partitions at depths 0, 1, 2, 3, and 5.
It is not part of ProverQA and must not be mixed into ProverQA RL training.

The conversion is deliberately fail-closed:

1. All 45,864 source `Unknown` questions are removed; they are never relabelled.
2. The only published options are `A) True` and `B) False`.
3. A means the query is derivable; B means its explicit negation is derivable.
4. Every retained label agrees with an independent forward-closure check.
5. The 38 source theories that are inconsistent under the repository's
   classical Z3 semantics are removed. This excludes 188 A/B questions and
   prevents an inconsistent premise set from vacuously proving every answer.
6. `nl2fol`, `conclusion_fol`, and the reference proof formulas use the prefix
   predicate syntax consumed by the repository parser, such as `sees cat dog`.
7. `reasoning` is extracted from the first official
   `proofsWithIntermediates` proof. Its FOL steps are retained in
   `evaluation_metadata.reference_reasoning_fol` for auditability.
8. Difficulty combines both source fields. QLen uses easy <= 2, medium 3-5,
   hard >= 6; QDep uses easy <= 1, medium 2-3, hard >= 4. The final class is
   the harder of the two bands.

The resulting full set contains 54,398 questions: 27,199 A and 27,199 B.

## Checked-in files

- `owa_binary_v1/pilot_300.jsonl`: 50 examples from each
  `(answer, difficulty)` group.
- `owa_binary_v1/benchmark_1500.jsonl`: 250 examples from each
  `(answer, difficulty)` group.
- `owa_binary_v1/manifest.json`: source hashes, conversion rules, full-set hash,
  counts, and checked-in output hashes.
- `owa_binary_v1/validation.json`: parser and Z3 validation results.

The reproducible full JSONL is about 106 MB and therefore is not checked in.
Its count and SHA-256 are recorded in the manifest.

## Reproduce

Run this from the repository root. `SOURCE_ROOT` must be the official OWA
directory that directly contains `depth-0` through `depth-5`.

```bash
# Point this at the official ProofWriter OWA directory.
SOURCE_ROOT=/path/to/proofwriter-dataset-V2020.12.3/OWA

# Generate the balanced checked-in subsets and the optional full set.
python scripts/convert_proofwriter_owa.py \
  --source-root "$SOURCE_ROOT" \
  --output-dir Data/ProofWriter/owa_binary_v1 \
  --split test \
  --pilot-size 300 \
  --benchmark-size 1500 \
  --seed 20260821 \
  --write-full

# Validate schema, prefix FOL, premise consistency, labels, and proof chains.
python scripts/validate_proofwriter_dataset.py \
  --report Data/ProofWriter/owa_binary_v1/validation.json \
  Data/ProofWriter/owa_binary_v1/pilot_300.jsonl \
  Data/ProofWriter/owa_binary_v1/benchmark_1500.jsonl
```

## Evaluate

The evaluator now recognizes this dataset from
`evaluation_metadata.dataset=ProofWriter`, so it will not report the run as
ProverQA. For example:

```bash
# Replace MODEL, ADAPTER, and RESULT_DIR with real paths on the GPU server.
python Test/Generation/evaluate.py \
  --dataset proofwriter \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --input Data/ProofWriter/owa_binary_v1/benchmark_1500.jsonl \
  --output_dir "$RESULT_DIR" \
  --name proofwriter_owa_binary_k3 \
  --num_samples 3
```

ProofWriter supplies derivation chains, but not the repository's manually
annotated canonical rule-name proof graphs. The evaluator therefore does not
require `canonical_proofs` for a detected ProofWriter run. Z3 metrics remain
available; rule-grounded metrics still measure generated steps against the
project's rule checker.
