"""Residual-stream activation capture.

`ActivationCapturer` loads a model with HuggingFace `transformers` and
`output_hidden_states=True`, then reduces each layer's token sequence to one
vector per prompt via a pluggable `Pooler`. The pooler is the polymorphic axis:
mean-pooling is a robust sentence-level default, but last-token pooling is the
natural choice for autoregressive next-token concepts, so both ship here.

Two non-negotiables for interpretability work, encoded as defaults:

1. **Never Ollama / llama.cpp.** Those serve text completions and expose no hook
   into the residual stream. Activation capture *requires* `transformers` with
   hidden states.
2. **Never quantize activations you intend to interpret.** 4-bit weights are fine
   for inference-only, but corrupt the residual-stream geometry a probe reads.
   We load in bf16 on GPU (float32 on CPU) and convert to float32 numpy.

`hidden_states[0]` is the embedding output (pre-transformer); layers `1..n` are
post-block residual streams. We keep all of them so the accuracy-by-layer curve
spans embedding to final layer, which is exactly what reveals whether a concept
is computed deep or merely present at the embedding.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Pooler(ABC):
    """Reduce a [batch, tokens, d_model] layer tensor to [batch, d_model]."""

    @abstractmethod
    def pool(self, hidden, attention_mask):  # types kept loose to avoid hard torch dep at import
        """Return a [batch, d_model] tensor. `hidden` and mask are torch tensors."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class MeanPooler(Pooler):
    """Mean over real (non-pad) tokens. Robust default for sentence concepts."""

    def pool(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        return (hidden * mask).sum(dim=1) / denom


class LastTokenPooler(Pooler):
    """Take the last real token's vector. Natural for next-token concepts."""

    def pool(self, hidden, attention_mask):
        # index of the last non-pad token per row
        lengths = attention_mask.sum(dim=1) - 1
        idx = lengths.long()
        batch = hidden.shape[0]
        rows = hidden[range(batch), idx]
        return rows


class ActivationCapturer:
    """Capture per-layer pooled residual activations for a list of prompts.

    Parameters
    ----------
    model_name:
        HF repo id. Defaults to a small, current model that fits an 8 GB GPU
        without quantization (`google/gemma-3-1b-it`, gated; accept its license
        on HuggingFace first).
    pooler:
        How to reduce tokens to one vector per prompt. Defaults to `MeanPooler`.
    batch_size:
        Prompts per forward pass. Lower it if you hit VRAM limits.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        pooler: Pooler | None = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.pooler = pooler or MeanPooler()
        self.batch_size = batch_size
        self._model = None
        self._tok = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            output_hidden_states=True,
        ).to(self._device)
        self._model.eval()

    def capture(self, texts: list[str]) -> dict[int, np.ndarray]:
        """Return {layer_index: [n_texts, d_model] float32} pooled activations."""
        import torch

        self._ensure_loaded()
        pooled_by_layer: dict[int, list[np.ndarray]] = {}
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                enc = self._tok(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(self._device)
                out = self._model(**enc)
                attn = enc["attention_mask"]
                for layer, hs in enumerate(out.hidden_states):
                    pooled = self.pooler.pool(hs, attn)
                    pooled_by_layer.setdefault(layer, []).append(
                        pooled.float().cpu().numpy()
                    )
        return {
            layer: np.concatenate(chunks, axis=0)
            for layer, chunks in pooled_by_layer.items()
        }

    # --- cache helpers: capture is the expensive step, so persist it ---------

    @staticmethod
    def save(acts_by_layer: dict[int, np.ndarray], labels: np.ndarray, path) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"layer_{k}": v for k, v in acts_by_layer.items()}
        arrays["labels"] = labels
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path):
        """Return (acts_by_layer, labels) from a .npz written by `save`."""
        data = np.load(path)
        acts = {
            int(k.split("_")[1]): data[k]
            for k in data.files if k.startswith("layer_")
        }
        return acts, data["labels"]
