"""Run single-shot structured generation and offline formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


FVCODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = FVCODE_ROOT / "Training"
for path in (FVCODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Test.Generation.metrics import (  # noqa: E402
    append_jsonl,
    attach_extended_metrics,
    evaluate_response,
    load_jsonl,
    plot_metrics_svg,
    summarize_by_difficulty,
    summarize_records,
    write_json,
    write_jsonl,
    write_report,
)
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (  # noqa: E402
    build_chat_prompt,
)
from recipe.formally_verifiable.rule_grounded_process_rl.rule_checker import RuleChecker  # noqa: E402
from recipe.formally_verifiable.common.verifier.z3_verifier import Z3Verifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model with one-shot structured generation."
    )
    parser.add_argument("--model", required=True, help="Base model path or HF id.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path. Omit to evaluate the base model.",
    )
    parser.add_argument("--input", required=True, help="Evaluation jsonl file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling; pass --no-do_sample for deterministic generation.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--z3_timeout_ms", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already completed (difficulty, problem_id, sample_index) records.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def stable_seed(base_seed: int, problem: dict[str, Any], sample_index: int) -> int:
    identity = f"{base_seed}|{problem.get('difficulty')}|{problem.get('id')}|{sample_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_chunk(
    *,
    model,
    tokenizer,
    prompt_text: str,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    device = model_device(model)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    return [
        tokenizer.decode(output[input_length:], skip_special_tokens=True)
        for output in outputs
    ]


def generate_responses(
    *,
    model,
    tokenizer,
    prompt_text: str,
    num_samples: int,
    generation_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[str]:
    responses: list[str] = []
    generation_batch_size = max(1, min(generation_batch_size, num_samples))
    while len(responses) < num_samples:
        current = min(generation_batch_size, num_samples - len(responses))
        if do_sample:
            responses.extend(
                generate_chunk(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    num_return_sequences=current,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            )
        else:
            for _ in range(current):
                responses.extend(
                    generate_chunk(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        num_return_sequences=1,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False,
                    )
                )
    return responses[:num_samples]


def record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("difficulty") or "unknown").strip().lower(),
        str(record.get("problem_id")),
        int(record.get("sample_index", 0) or 0),
    )


def load_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    incremental = output_dir / "records" / "incremental_records.jsonl"
    final = output_dir / "records.jsonl"
    if incremental.exists():
        return load_jsonl(incremental)
    if final.exists():
        return load_jsonl(final)
    return []


def write_response_text(output_dir: Path, record: dict[str, Any]) -> None:
    response_dir = output_dir / "responses" / "text"
    response_dir.mkdir(parents=True, exist_ok=True)
    difficulty = str(record.get("difficulty") or "unknown")
    problem_id = str(record.get("problem_id"))
    sample_index = int(record.get("sample_index", 0) or 0)
    safe = f"{difficulty}_problem_{problem_id}_sample_{sample_index}".replace("/", "_")
    (response_dir / f"{safe}.txt").write_text(
        str(record.get("response", "") or ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(parents=True, exist_ok=True)
    (output_dir / "responses" / "text").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    run_name = args.name or (Path(args.adapter).name if args.adapter else Path(args.model).name)
    problems = load_jsonl(args.input, args.max_samples)
    existing_records = load_existing_records(output_dir) if args.resume else []
    completed = {record_key(record) for record in existing_records}
    records = list(existing_records)

    model, tokenizer = load_model_and_tokenizer(args)
    verifier = Z3Verifier(timeout_ms=args.z3_timeout_ms)
    rule_checker = RuleChecker(timeout_ms=args.z3_timeout_ms)

    for problem in tqdm(problems, desc=f"Evaluate {run_name}"):
        prompt_text = build_chat_prompt(tokenizer, problem)
        pending_sample_indices = [
            index
            for index in range(args.num_samples)
            if (
                str(problem.get("difficulty") or "unknown").strip().lower(),
                str(problem.get("id")),
                index,
            )
            not in completed
        ]
        if not pending_sample_indices:
            continue

        for start in range(0, len(pending_sample_indices), max(1, args.generation_batch_size)):
            sample_indices = pending_sample_indices[start : start + max(1, args.generation_batch_size)]
            set_seed(stable_seed(args.seed, problem, sample_indices[0]))
            responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                num_samples=len(sample_indices),
                generation_batch_size=len(sample_indices),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            for sample_index, response in zip(sample_indices, responses):
                metrics = evaluate_response(
                    response,
                    problem,
                    verifier=verifier,
                    rule_checker=rule_checker,
                )
                record = {
                    "method_type": "generation",
                    "run_name": run_name,
                    "problem_id": problem.get("id"),
                    "difficulty": problem.get("difficulty"),
                    "ground_truth": problem.get("answer"),
                    "sample_index": sample_index,
                    "response": response,
                    "metrics": metrics,
                }
                record = attach_extended_metrics(record, problem)
                records.append(record)
                completed.add(record_key(record))
                append_jsonl(output_dir / "records" / "incremental_records.jsonl", record)
                write_response_text(output_dir, record)

    k = min(3, args.num_samples)
    summary = {
        "run_name": run_name,
        "method_type": "generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": args.adapter,
        "input": args.input,
        "output_dir": str(output_dir),
        "max_samples": args.max_samples,
        "num_samples": args.num_samples,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "z3_timeout_ms": args.z3_timeout_ms,
        "prompt_source": (
            "recipe.formally_verifiable.rule_grounded_process_rl."
            "structured_prompt.build_chat_prompt"
        ),
        "inference_protocol": "single_shot_full_structured_generation",
        "metrics": summarize_records(records, k=k),
        "by_difficulty": summarize_by_difficulty(records, k=k),
    }
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_config.json", {key: value for key, value in vars(args).items()})
    write_report(output_dir / "report.md", summary)
    plot_metrics_svg(output_dir / "plots" / "metrics.svg", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
