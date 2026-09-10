from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from Test.Generation.metrics import evaluation_dataset_protocol


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "Data" / "ProofWriter" / "owa_binary_v1"
BINARY_OPTIONS = ["A) True", "B) False"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_checked_in_outputs_match_manifest_and_binary_policy() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name in ("pilot_300.jsonl", "benchmark_1500.jsonl"):
        path = DATA_ROOT / name
        rows = load_jsonl(path)
        expected = manifest["outputs"][name]
        assert len(rows) == expected["records"]
        assert sha256(path) == expected["sha256"]
        assert len({row["id"] for row in rows}) == len(rows)
        assert all(row["options"] == BINARY_OPTIONS for row in rows)
        assert all(row["answer"] in {"A", "B"} for row in rows)
        assert all("unknown" not in " ".join(row["options"]).lower() for row in rows)


def test_balanced_benchmark_and_joint_difficulty_metadata() -> None:
    rows = load_jsonl(DATA_ROOT / "benchmark_1500.jsonl")
    counts = Counter((row["answer"], row["difficulty"]) for row in rows)
    assert set(counts.values()) == {250}

    rank = {"easy": 0, "medium": 1, "hard": 2}
    for row in rows:
        metadata = row["evaluation_metadata"]
        qlen = metadata["QLen"]
        qdep = metadata["QDep"]
        qlen_band = "easy" if qlen <= 2 else "medium" if qlen <= 5 else "hard"
        qdep_band = "easy" if qdep <= 1 else "medium" if qdep <= 3 else "hard"
        expected = max((qlen_band, qdep_band), key=rank.__getitem__)
        assert row["difficulty"] == expected
        assert metadata["dataset"] == "ProofWriter"
        assert metadata["proof_source"] == "proofsWithIntermediates"
        assert metadata["reference_reasoning_fol"]


def test_evaluator_detects_proofwriter_without_canonical_graphs() -> None:
    problems = load_jsonl(DATA_ROOT / "pilot_300.jsonl")[:2]
    protocol = evaluation_dataset_protocol(problems)
    assert protocol["dataset"] == "proofwriter"
    assert protocol["canonical_proof_required"] is False
    assert protocol["subset"] == "AB_unknown_excluded_z3_verified"
