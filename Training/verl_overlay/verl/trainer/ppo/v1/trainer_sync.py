# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

import ray

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        if self._load_lora_adapter_from_checkpoint(prefer_resume_dir=True):
            return
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            if self._is_checkpoint_step() and self._load_lora_adapter_from_checkpoint(prefer_resume_dir=False):
                return
            skip_on_save_step = self.config.trainer.get("skip_update_weights_on_save_step", False)
            save_freq = self.config.trainer.get("save_freq", -1)
            if skip_on_save_step and save_freq > 0 and self.global_steps > 0 and self.global_steps % save_freq == 0:
                logger.warning("Skipping update_weights at checkpoint step %s", self.global_steps)
                return
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()

    def _is_checkpoint_step(self) -> bool:
        save_freq = self.config.trainer.get("save_freq", -1)
        return save_freq > 0 and self.global_steps > 0 and self.global_steps % save_freq == 0

    def _load_lora_adapter_from_checkpoint(self, prefer_resume_dir: bool) -> bool:
        if not self.config.trainer.get("load_lora_adapter_from_checkpoint", False):
            return False
        if self.global_steps is None or self.global_steps <= 0:
            return False

        adapter_path = self._resolve_lora_adapter_path(self.global_steps, prefer_resume_dir=prefer_resume_dir)
        if adapter_path is None:
            message = f"No valid LoRA adapter checkpoint found for global_step_{self.global_steps}"
            if self.config.trainer.get("require_lora_adapter_checkpoint", False):
                raise FileNotFoundError(message)
            logger.warning("%s; falling back to actor tensor update_weights", message)
            return False

        logger.info("Loading rollout LoRA adapter from %s for global_step_%s", adapter_path, self.global_steps)
        ray.get(self.actor_rollout_wg.load_lora_adapter(lora_path=adapter_path, global_steps=self.global_steps))
        return True

    def _resolve_lora_adapter_path(self, global_steps: int, prefer_resume_dir: bool) -> str | None:
        base_dirs = []
        resume_dir = self.config.trainer.get("resume_lora_adapter_dir", None)
        checkpoint_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        export_dir = checkpoint_cfg.get("export_lora_adapter_dir", None)

        if prefer_resume_dir and resume_dir:
            base_dirs.append(resume_dir)
        if export_dir:
            base_dirs.append(export_dir)
        if not prefer_resume_dir and resume_dir:
            base_dirs.append(resume_dir)

        seen = set()
        for base_dir in base_dirs:
            if not base_dir or base_dir in seen:
                continue
            seen.add(base_dir)
            adapter_path = os.path.join(str(base_dir), f"global_step_{global_steps}")
            if self._is_valid_lora_adapter_path(adapter_path):
                return adapter_path
        return None

    @staticmethod
    def _is_valid_lora_adapter_path(adapter_path: str) -> bool:
        adapter_config = os.path.join(adapter_path, "adapter_config.json")
        adapter_model = os.path.join(adapter_path, "adapter_model.safetensors")
        return (
            os.path.isdir(adapter_path)
            and os.path.isfile(adapter_config)
            and os.path.isfile(adapter_model)
            and os.path.getsize(adapter_model) > 40
        )
