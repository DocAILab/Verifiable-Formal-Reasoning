from __future__ import annotations

import sys
from pathlib import Path

import pytest


FVCODE_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = FVCODE_ROOT / "Training"
for path in (FVCODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Training.Warmup.train_lora import prepare_response_target
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (
    build_chat_prompt,
)


RESPONSE = "<think>reasoning</think>\n<summary>[]</summary>"


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "prompt"


def test_explicit_target_keeps_think_wrapper() -> None:
    assert prepare_response_target(RESPONSE, "explicit") == RESPONSE


def test_native_target_drops_only_duplicate_opening_think() -> None:
    target = prepare_response_target(RESPONSE, "native")
    assert not target.startswith("<think>")
    assert target == "reasoning</think>\n<summary>[]</summary>"


@pytest.mark.parametrize(
    ("mode", "enabled"), [("explicit", False), ("native", True)]
)
def test_prompt_switches_native_thinking(mode: str, enabled: bool) -> None:
    tokenizer = FakeTokenizer()
    problem = {"nl2fol": {}, "question": "Q", "options": [], "conclusion_fol": "p"}
    assert build_chat_prompt(tokenizer, problem, thinking_mode=mode) == "prompt"
    assert tokenizer.kwargs["enable_thinking"] is enabled
    assert ("native thinking mode" in tokenizer.messages[0]["content"]) is enabled


def test_rejects_malformed_targets() -> None:
    with pytest.raises(ValueError):
        prepare_response_target("<summary>[]</summary>", "explicit")
    with pytest.raises(ValueError):
        prepare_response_target("reasoning only", "native")
