"""Bounded reference-weight offload for single-device, full-parameter GRPO."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from trl import GRPOTrainer


class ReferenceOffloader:
    def __init__(self, model, budget_gib: float, execution_device):
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("CPU offload is only supported for a frozen reference model")
        self.device = torch.device(execution_device)
        self.budget_bytes = int(budget_gib * 1024**3)
        self.layers = []
        self.weight_bytes = 0
        self.cycles = 0
        self.transfer_seconds = 0.0
        self.active = False
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Reference offload requires a model.model.layers decoder")
        # Skip tied parameters shared with other blocks or embeddings. Moving one
        # such tensor would also move the unselected module that shares it.
        ownership = {}
        for parameter in model.parameters():
            ownership[id(parameter)] = 0
        for module in model.modules():
            for parameter in module.parameters(recurse=False):
                ownership[id(parameter)] += 1
        for index, layer in reversed(list(enumerate(layers))):
            tensors = list(layer.parameters()) + list(layer.buffers())
            if any(ownership.get(id(parameter), 0) > 1 for parameter in layer.parameters()):
                continue
            size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            if size and self.weight_bytes + size <= self.budget_bytes:
                self.layers.append((f"model.layers.{index}", layer))
                self.weight_bytes += size
        if not self.layers:
            raise ValueError("Reference CPU budget is too small to offload one decoder layer")
        self._move("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _move(self, device):
        started = time.perf_counter()
        for _, layer in self.layers:
            layer.to(device)
        self.transfer_seconds += time.perf_counter() - started

    @contextmanager
    def on_execution_device(self):
        if self.active:
            raise RuntimeError("Reference offload contexts cannot be nested")
        self.active = True
        try:
            self._move(self.device)
            # All reference microbatches run before the selected layers are moved
            # back. There is one transfer cycle per rollout batch, not per answer.
            yield
            self.cycles += 1
        finally:
            try:
                self._move("cpu")
                if self.device.type == "cuda":
                    # Reference copies are temporary. Release their unused GPU
                    # allocations before policy backward needs large workspaces.
                    torch.cuda.empty_cache()
            finally:
                self.active = False

    def statistics(self):
        return {
            "budget_gib": self.budget_bytes / 1024**3,
            "offloaded_gib": round(self.weight_bytes / 1024**3, 4),
            "offloaded_bytes": self.weight_bytes,
            "layer_names": [name for name, _ in self.layers],
            "reference_batches": self.cycles,
            "transfer_seconds": round(self.transfer_seconds, 3),
        }


class OffloadGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, reference_cpu_offload_gib=0.0, trim_training_padding=False,
                 **kwargs):
        self.reference_offloader = None
        self.trim_training_padding = trim_training_padding
        super().__init__(*args, **kwargs)
        if reference_cpu_offload_gib and self.ref_model is not None:
            if self.is_deepspeed_enabled or self.is_fsdp_enabled or self.args.sync_ref_model:
                raise ValueError(
                    "Reference offload requires single-device GRPO with a fixed reference"
                )
            self.ref_model.requires_grad_(False)
            self.reference_offloader = ReferenceOffloader(
                self.ref_model, reference_cpu_offload_gib, self.accelerator.device
            )
            print(f"Reference CPU offload: {self.reference_offloader.statistics()}", flush=True)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = model if model is not None else self.model
        if target.config.model_type != "lfm3":
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        # LFM3 saves g_proj under gate_proj. Its native loader converts those
        # names; Trainer's raw state_dict loader otherwise skips the gate weights.
        restored = target.__class__.from_pretrained(
            resume_from_checkpoint, dtype=next(target.parameters()).dtype,
            device_map="cpu", attn_implementation="sdpa",
        )
        target.load_state_dict(restored.state_dict(), strict=True)
        del restored

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        if self.reference_offloader is not None and model is self.ref_model:
            with self.reference_offloader.on_execution_device():
                return self._token_stats(model, *args, **kwargs)
        return self._token_stats(model, *args, **kwargs)

    def _token_stats(self, model, input_ids, attention_mask, logits_to_keep,
                     batch_size=None, **kwargs):
        # Only trim masked suffixes for ordinary causal text models. Keep the
        # parent's handling of multimodal fields, sampling supports and MoE loss.
        supported = {"compute_entropy", "compute_aux_loss"}
        if (not self.trim_training_padding or logits_to_keep <= 0
                or kwargs.get("compute_aux_loss", False)
                or any(value is not None for key, value in kwargs.items()
                       if key not in supported)):
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, **kwargs,
            )
        prompt_length = input_ids.shape[1] - logits_to_keep
        positions = torch.arange(1, logits_to_keep + 1, device=attention_mask.device)
        # One host transfer per call, rather than a CUDA sync for every row.
        keep = (attention_mask[:, prompt_length:].ne(0) * positions).amax(dim=1)
        keep = keep.clamp(min=1).tolist()
        batch_size = batch_size or input_ids.shape[0]
        logps, entropies = [], []
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            length = max(keep[start:end])
            stop = prompt_length + length
            logp, entropy, _ = super()._get_per_token_logps_and_entropies(
                model, input_ids[start:end, :stop], attention_mask[start:end, :stop],
                length, batch_size=None, **kwargs,
            )
            # Restore the original rollout shape so masks, KL, group advantages
            # and DAPO's global token denominator are unchanged. F.pad retains
            # gradients for the actual generated tokens.
            padding = (0, logits_to_keep - length)
            logps.append(torch.nn.functional.pad(logp, padding))
            if entropy is not None:
                entropies.append(torch.nn.functional.pad(entropy, padding))
        return (torch.cat(logps), torch.cat(entropies) if entropies else None, None)

    def offload_statistics(self):
        return self.reference_offloader.statistics() if self.reference_offloader else {}
