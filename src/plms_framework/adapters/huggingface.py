from __future__ import annotations

import importlib.util

from ..errors import DependencyUnavailableError, ModelUnavailableError, ValidationError
from ..models import Availability, ProteinSequence, RawEmbedding
from .base import ModelAdapter


class HuggingFaceAdapter(ModelAdapter):
    """Lazy adapter for compatible Hugging Face encoder checkpoints."""

    def __init__(self, manifest) -> None:
        super().__init__(manifest)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._resolved_revision = None

    def availability(self) -> Availability:
        missing = [
            name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return Availability(
                False, "dependency_missing", f"Install dependencies: {', '.join(missing)}"
            )
        return Availability(True)

    def load(self, *, device: str = "auto", precision: str = "auto") -> None:
        availability = self.availability()
        if not availability.available:
            raise DependencyUnavailableError(
                availability.detail or "Hugging Face dependencies unavailable"
            )
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float32 if precision == "float32" else None
        if precision in {"float16", "fp16"}:
            dtype = torch.float16
        elif precision in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        kwargs = {"revision": self.manifest.revision, "trust_remote_code": False}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.manifest.artifact, revision=self.manifest.revision
            )
            model_class = (
                AutoModelForSeq2SeqLM
                if self.manifest.architecture == "encoder-decoder"
                else AutoModelForMaskedLM
            )
            loaded = model_class.from_pretrained(self.manifest.artifact, **kwargs).to(device).eval()
            self._resolved_revision = getattr(loaded.config, "_commit_hash", None)
            self._model = (
                loaded.get_encoder()
                if self.manifest.architecture == "encoder-decoder"
                else loaded.base_model
            )
            self._dtype = str(next(self._model.parameters()).dtype)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load {self.manifest.id}: {exc}",
                context={"artifact": self.manifest.artifact},
            ) from exc
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._resolved_revision = None

    def provenance(self) -> dict:
        value = super().provenance()
        value["resolved_revision"] = self._resolved_revision
        value["device"] = self._device
        value["dtype"] = self._dtype
        return value

    def _model_input(self, sequence: str) -> str:
        if self.manifest.preprocess == "space_separated":
            return " ".join(sequence)
        return sequence

    def embed(
        self,
        sequences: tuple[ProteinSequence, ...],
        *,
        layer: str | int = "model_default",
        device: str = "auto",
        precision: str = "auto",
    ) -> list[RawEmbedding]:
        self.load(device=device, precision=precision)
        import torch

        inputs = [self._model_input(item.sequence) for item in sequences]
        encoded = self._tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask")
        model_inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output = self._model(**model_inputs, output_hidden_states=isinstance(layer, int))
        hidden = output.hidden_states[layer] if isinstance(layer, int) else output.last_hidden_state
        output_dtype = str(hidden.dtype)
        hidden = hidden.detach().float().cpu()
        attention = encoded["attention_mask"].cpu()
        results = []
        for row, item in enumerate(sequences):
            residue_mask = (attention[row] == 1) & (special[row] == 0)
            residue_tensor = hidden[row][residue_mask]
            if residue_tensor.shape[0] != len(item.sequence):
                raise ValidationError(
                    f"Tokenizer mapping for {self.manifest.id} produced {residue_tensor.shape[0]} residue tokens "
                    f"for {len(item.sequence)} residues",
                    context={"sequence_id": item.id},
                )
            native = None
            if "pooler_output" in output and output.pooler_output is not None:
                native = output.pooler_output[row].detach().float().cpu().tolist()
            results.append(
                RawEmbedding(
                    item.id,
                    residue_tensor.tolist(),
                    native=native,
                    token_to_residue=list(range(1, len(item.sequence) + 1)),
                    metadata={"layer": layer, "device": self._device, "dtype": output_dtype},
                )
            )
        return results
