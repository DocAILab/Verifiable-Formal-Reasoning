# FOLIO Evaluation Data

## Files and Scope

- `converted_train.jsonl` and `converted_validation.jsonl`: unchanged local
  source conversions. Do not pass them directly to the structured evaluator.
- `verified_ab_v1/validation.jsonl`: recommended cross-dataset evaluation input.
- `verified_ab_v1/train.jsonl`: separately prepared original training split;
  preparation does not start training or add it to the ProverQA training set.
- `verified_ab_v1/manifest.json`: source/output SHA-256, preparation and Z3
  versions, label counts, exclusion counts and protocol notes.
- `verified_ab_v1/excluded_*.jsonl`: original IDs, source line numbers, labels
  and concrete exclusion reasons. Original excluded examples remain in raw files.

Current reproducible counts (Z3 4.16.0, 5000 ms per solver check):

| Split | Original | Uncertain removed | A/B candidates | Accepted A | Accepted B | Accepted total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1001 | 324 | 677 | 146 | 93 | 239 |
| Validation | 203 | 69 | 134 | 27 | 24 | 51 |

| Split | Unsupported/malformed formula | Gold label not entailed | Inconsistent premises |
|---|---:|---:|---:|
| Train | 268 | 168 | 2 |
| Validation | 70 | 13 | 0 |

These are first-failure counts, not independent error frequencies. Exclusions
include parser limitations, not just bad source annotations. No solver-unknown
rows were observed in this build; future unknown/timeout results are excluded.

## Preparation Policy

Run from the repository root:

```bash
python Test/Generation/prepare_folio.py
```

1. Keep the original split and A/B labels. Exclude gold C, but retain all three
   answer options so the shared prompt stays consistent with previous experiments.
2. Extract only the context premise block. Split its NL and FOL lines one to one,
   and require matching counts and unique NL keys. The query mapping is not a
   premise; a genuine context premise identical to the query is still retained.
3. Normalize atomic `P(a, b)` to prefix form using tokens and the existing AST,
   preserving logical grouping and explicit quantifier-body scope. Check types,
   bound variables and predicate arity.
   Do not guess missing parentheses, repair names, add quantifiers, reinterpret
   equality as a predicate, or resolve ambiguous implication chains.
4. Require satisfiable premises P, then UNSAT(P AND NOT Q) for A or
   UNSAT(P AND Q) for B. Never accept inconsistent premises by explosion;
   timeout/unknown is not a verification success. Never relabel to fit Z3.
5. Remove fake zero-length proof metadata. Store provenance and explicit
   reference-proof/difficulty availability metadata in each accepted row.

Equality, function terms and some quantifier syntax remain unsupported by the
current shared parser. This preparation intentionally does not rewrite the
training parser or expand the model's rule ontology. Surviving existential
formulas can be checked by Z3, but existential rules are not covered by the
closed RuleChecker ontology (19/51 validation and 93/239 train problems contain
existentials). Each row records this feature for later analysis.

## Interpretation

- Evaluate existing ProverQA-trained adapters on the 51 validation examples for
  a **curated supported A/B cross-dataset diagnostic**, not a full FOLIO benchmark.
  The small sample and filtering bias limit conclusions. Do not merge the 239
  original training examples into this test set or call it a 290-example test.
- Z3 validates the supplied formulas against the labels, not the fidelity of the
  original natural-language-to-FOL translation. Manual semantic review is still
  needed, especially for source symbol inconsistencies that do not change labels.
- RGD is unavailable because there are no reference proofs. JSON uses `null`,
  reports use `N/A`; neither a fabricated one-step proof nor a zero score is used.
- FOLIO has no local Easy/Medium/Hard annotation. Results are Overall plus
  `unknown` (unannotated), not artificially inferred difficulty levels.
- Z3(no cascade) is semantic-only. Rule-grounded metrics additionally use the
  unchanged training RuleChecker and are not a complete proof-validity metric
  for unrestricted FOLIO. Report this limitation alongside those scores.

See [`Test/README.md`](../../Test/README.md) for evaluation and four-GPU commands.
