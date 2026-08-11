import os
from typing import Any


def _active_adapter_name(peft_model: Any) -> str:
    adapter_name = getattr(peft_model, "active_adapter", None)
    if callable(adapter_name):
        adapter_name = adapter_name()
    if isinstance(adapter_name, (list, tuple)):
        adapter_name = adapter_name[0] if adapter_name else None
    return adapter_name or "default"


def _with_adapter_name_for_peft_save(state_dict: dict[str, Any], adapter_name: str) -> dict[str, Any]:
    """PEFT save_pretrained filters by adapter name even when given a state_dict.

    get_peft_model_state_dict returns LoRA keys with the adapter name stripped,
    e.g. ``lora_A.weight``. Passing those keys back into save_pretrained causes
    PEFT to filter everything out. Reinsert the active adapter segment for save.
    """
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
            stripped = f"{marker}weight"
            with_adapter = f"{marker}{adapter_name}.weight"
            if stripped in new_key and with_adapter not in new_key:
                new_key = new_key.replace(stripped, with_adapter)
        normalized[new_key] = value
    return normalized


def _has_lora_tensors(state_dict: dict[str, Any]) -> bool:
    for key, value in state_dict.items():
        if "lora_" not in key:
            continue
        numel = getattr(value, "numel", None)
        if numel is None or numel() > 0:
            return True
    return False


def save_lora_adapter_checkpoint(
    *,
    export_root: str,
    global_step: int,
    peft_model: Any,
    state_dict: dict[str, Any],
) -> str | None:
    if not export_root:
        return None
    if not _has_lora_tensors(state_dict):
        raise ValueError(f"Refusing to export empty LoRA adapter state_dict at global_step_{global_step}.")

    state_dict = _with_adapter_name_for_peft_save(state_dict, _active_adapter_name(peft_model))
    export_path = os.path.join(export_root, f"global_step_{global_step}")
    os.makedirs(export_path, exist_ok=True)
    peft_model.save_pretrained(export_path, state_dict=state_dict, safe_serialization=True)
    adapter_file = os.path.join(export_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file) and os.path.getsize(adapter_file) <= 40:
        raise ValueError(f"PEFT wrote an empty LoRA adapter checkpoint at global_step_{global_step}.")
    return export_path
