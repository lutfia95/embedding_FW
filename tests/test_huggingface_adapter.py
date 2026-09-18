import sys
import types
import unittest
from unittest.mock import patch

from plms_framework.models import Availability
from plms_framework.registry import default_registry


class _FakeBaseModel:
    def parameters(self):
        yield types.SimpleNamespace(dtype="float32")


class _FakeLoadedModel:
    def __init__(self):
        self.config = types.SimpleNamespace(_commit_hash="resolved-revision")
        self.base_model = _FakeBaseModel()
        self.encoder = _FakeBaseModel()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def get_encoder(self):
        return self.encoder


class _Loader:
    calls = []

    @classmethod
    def from_pretrained(cls, artifact, **kwargs):
        cls.calls.append((artifact, kwargs))
        return _FakeLoadedModel()


class _AutoModel(_Loader):
    calls = []


class _AutoModelForSeq2SeqLM(_Loader):
    calls = []


class _AutoModelForCausalLM(_Loader):
    calls = []


class _FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = None


class _AutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, artifact, **kwargs):
        cls.calls.append((artifact, kwargs))
        return _FakeTokenizer()


class HuggingFaceAdapterLoadingTests(unittest.TestCase):
    def setUp(self):
        for loader in (
            _AutoModel,
            _AutoModelForSeq2SeqLM,
            _AutoModelForCausalLM,
            _AutoTokenizer,
        ):
            loader.calls.clear()
        self.fake_torch = types.ModuleType("torch")
        self.fake_transformers = types.ModuleType("transformers")
        self.fake_transformers.AutoModel = _AutoModel
        self.fake_transformers.AutoModelForSeq2SeqLM = _AutoModelForSeq2SeqLM
        self.fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
        self.fake_transformers.AutoTokenizer = _AutoTokenizer

    def _load(self, model_id):
        adapter = default_registry.adapter(model_id)
        with (
            patch.dict(
                sys.modules,
                {"torch": self.fake_torch, "transformers": self.fake_transformers},
            ),
            patch.object(adapter, "availability", return_value=Availability(True)),
        ):
            adapter.load(device="cpu", precision="auto")
        return adapter

    def test_encoder_uses_auto_model(self):
        adapter = self._load("protxlnet")
        self.assertEqual(len(_AutoModel.calls), 1)
        self.assertEqual(len(_AutoModelForSeq2SeqLM.calls), 0)
        self.assertEqual(adapter.provenance()["resolved_revision"], "resolved-revision")

    def test_encoder_decoder_uses_seq2seq_encoder(self):
        adapter = self._load("prott5-xl-bfd")
        self.assertEqual(len(_AutoModelForSeq2SeqLM.calls), 1)
        self.assertIsInstance(adapter._model, _FakeBaseModel)

    def test_causal_model_uses_remote_code_only_when_declared(self):
        adapter = self._load("progen2-small")
        self.assertEqual(len(_AutoModelForCausalLM.calls), 1)
        _, model_kwargs = _AutoModelForCausalLM.calls[0]
        _, tokenizer_kwargs = _AutoTokenizer.calls[0]
        self.assertTrue(model_kwargs["trust_remote_code"])
        self.assertTrue(tokenizer_kwargs["trust_remote_code"])
        self.assertTrue(adapter.provenance()["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
