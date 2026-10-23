from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys

from ..errors import DependencyUnavailableError, ModelUnavailableError, ValidationError
from ..models import Availability, ProteinSequence, RawEmbedding
from .base import ModelAdapter

_MINIMUM_ESM_VERSION = (3, 4, 1, 1)


def _version_tuple(value: str) -> tuple[int, ...]:
    release = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if release is None:
        return ()
    post = re.search(r"\.post(\d+)", value)
    return (*map(int, release.groups()), int(post.group(1)) if post else 0)


class ESMCAdapter(ModelAdapter):
    """Lazy adapter for the maintained Biohub ESM-C implementation."""

    def __init__(self, manifest) -> None:
        super().__init__(manifest)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._esm_version = None

    def availability(self) -> Availability:
        if sys.version_info < (3, 12):
            return Availability(
                False,
                "python_version_unsupported",
                "Current ESM-C support requires Python 3.12 or newer; "
                "use a separate Python 3.12+ environment and install plms-framework[esmc]",
            )
        missing = [
            name for name in ("torch", "esm") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return Availability(
                False,
                "dependency_missing",
                "Install the ESM-C backend with: pip install 'plms-framework[esmc]'",
            )
        try:
            version = importlib.metadata.version("esm")
        except importlib.metadata.PackageNotFoundError:
            return Availability(
                False,
                "dependency_incompatible",
                "An 'esm' module was found, but the Biohub 'esm' distribution was not. "
                "Remove conflicting packages such as fair-esm and install plms-framework[esmc]",
            )
        if _version_tuple(version) < _MINIMUM_ESM_VERSION:
            return Availability(
                False,
                "dependency_incompatible",
                f"Biohub esm {version} is too old; install esm>=3.4.1.post1,<4",
            )
        return Availability(True)

    def load(self, *, device: str = "auto", precision: str = "auto") -> None:
        availability = self.availability()
        if not availability.available:
            raise DependencyUnavailableError(availability.detail or "ESM-C backend unavailable")
        if self._model is not None:
            return

        import torch
        from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError(
                f"Could not load {self.manifest.id}: CUDA was requested but is unavailable",
                context={"artifact": self.manifest.artifact, "device": device},
            )

        resolved_precision = precision
        if resolved_precision == "auto":
            resolved_precision = "bfloat16" if device == "cuda" else "float32"
        if resolved_precision == "bf16":
            resolved_precision = "bfloat16"
        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[resolved_precision]

        try:
            self._model = EsmcForMaskedLM.from_pretrained(
                self.manifest.artifact,
                revision=self.manifest.revision,
                device=device,
                dtype=dtype,
                # SDPA is portable across the advertised CPU and CUDA paths.
                # Fused kernels can be selected in a future explicit option.
                attn_implementation="sdpa",
            ).eval()
            self._tokenizer = EsmcTokenizer()
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise ModelUnavailableError(
                f"Could not load {self.manifest.id}: {exc}",
                context={
                    "artifact": self.manifest.artifact,
                    "device": device,
                    "precision": resolved_precision,
                },
            ) from exc
        self._device = device
        self._dtype = str(next(self._model.parameters()).dtype)
        self._esm_version = importlib.metadata.version("esm")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._esm_version = None

    def provenance(self) -> dict:
        value = super().provenance()
        value.update(
            {
                "device": self._device,
                "dtype": self._dtype,
                "esm_version": self._esm_version,
                "attention_implementation": "sdpa" if self._model is not None else None,
            }
        )
        return value

    def estimate_resources(self, sequences) -> dict:
        value = super().estimate_resources(sequences)
        # Published ESM-C checkpoints are currently stored as float32 weights.
        value["checkpoint_bytes_estimate"] = (self.manifest.parameters or 0) * 4
        return value

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

        inputs = [item.sequence for item in sequences]
        encoded = self._tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask").cpu()
        attention = encoded["attention_mask"].cpu()
        model_inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            output = self._model(
                **model_inputs,
                output_hidden_states=isinstance(layer, int),
                return_dict=True,
            )
        if isinstance(layer, int):
            if output.hidden_states is None:
                raise ModelUnavailableError(
                    f"{self.manifest.id} did not return requested hidden states"
                )
            hidden = output.hidden_states[layer]
        else:
            hidden = output.last_hidden_state
        if hidden is None:
            raise ModelUnavailableError(f"{self.manifest.id} did not return embeddings")

        output_dtype = str(hidden.dtype)
        hidden = hidden.detach().float().cpu()
        results = []
        for row, item in enumerate(sequences):
            residue_mask = (attention[row] == 1) & (special[row] == 0)
            residue_tensor = hidden[row][residue_mask]
            if residue_tensor.shape[0] != len(item.sequence):
                raise ValidationError(
                    f"Tokenizer mapping for {self.manifest.id} produced "
                    f"{residue_tensor.shape[0]} residue tokens for {len(item.sequence)} residues",
                    context={"sequence_id": item.id},
                )
            results.append(
                RawEmbedding(
                    sequence_id=item.id,
                    residues=residue_tensor.tolist(),
                    token_to_residue=list(range(1, len(item.sequence) + 1)),
                    metadata={
                        "layer": layer,
                        "device": self._device,
                        "dtype": output_dtype,
                    },
                )
            )
        return results
