# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from dataclasses import MISSING, FrozenInstanceError, fields
from types import ModuleType

import pytest

from sglang_omni.models.model_capabilities import (
    ModelCapabilities,
    PrefillCudaGraphBackendCapability,
    PrefillCudaGraphCapability,
    get_model_capabilities,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

EXPECTED_TTS_CAPABILITIES = {
    "DotsTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=False,
        supports_torch_compile=True,
    ),
    "AudarTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=False,
        supports_torch_compile=False,
    ),
    "Qwen3TTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
    ),
    "HiggsMultimodalQwen3ForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        prefill_cuda_graph=PrefillCudaGraphCapability(
            integration="incompatible",
            incompatible_reason=(
                "padded prefill changes Higgs model outputs; keep prefill eager"
            ),
        ),
    ),
    "MossTTSDelayModel": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=False,
    ),
    "MossTTSLocalModel": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
    ),
    "FishQwen3OmniForCausalLM": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
    ),
    "BailingMMNativeForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=False,
        supports_cuda_graph=True,
        supports_torch_compile=False,
    ),
    "VoxtralTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=False,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=True,
        supports_torch_compile=True,
    ),
    "Zonos2ForCausalLM": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
    ),
}


def _package_for_architecture(architecture: str):
    config_cls = PIPELINE_CONFIG_REGISTRY.configs.get(architecture)
    assert config_cls is not None, f"{architecture} is not registered"
    return importlib.import_module(config_cls.__module__.rsplit(".", 1)[0])


def _capability_required_architectures() -> set[str]:
    return {
        config_cls.architecture
        for config_cls in set(PIPELINE_CONFIG_REGISTRY.configs.values())
        if getattr(config_cls, "requires_model_capabilities", False)
    }


def test_expected_capabilities_cover_registered_required_configs() -> None:
    assert _capability_required_architectures() == set(EXPECTED_TTS_CAPABILITIES)


def test_required_model_capability_configs_resolve_capabilities() -> None:
    for architecture in sorted(_capability_required_architectures()):
        assert get_model_capabilities(architecture) is not None


def test_model_capabilities_are_frozen_and_explicit() -> None:
    boolean_fields = fields(ModelCapabilities)[:-1]
    for field in boolean_fields:
        assert field.type in (bool, "bool")
        assert field.default is MISSING
        assert field.default_factory is MISSING
    prefill_field = fields(ModelCapabilities)[-1]
    assert prefill_field.name == "prefill_cuda_graph"
    assert prefill_field.default is None

    with pytest.raises(TypeError):
        ModelCapabilities()

    capabilities = next(iter(EXPECTED_TTS_CAPABILITIES.values()))
    with pytest.raises(FrozenInstanceError):
        capabilities.supports_reference_audio = False


def _capabilities_with_prefill(
    capability: PrefillCudaGraphCapability,
) -> ModelCapabilities:
    return ModelCapabilities(
        supports_reference_audio=False,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=False,
        supports_torch_compile=False,
        prefill_cuda_graph=capability,
    )


def test_only_higgs_declares_prefill_cuda_graph_policy() -> None:
    assert all(
        capabilities.prefill_cuda_graph is None
        for architecture, capabilities in EXPECTED_TTS_CAPABILITIES.items()
        if architecture != "HiggsMultimodalQwen3ForConditionalGeneration"
    )
    higgs = EXPECTED_TTS_CAPABILITIES["HiggsMultimodalQwen3ForConditionalGeneration"]
    assert higgs.prefill_cuda_graph is not None
    assert higgs.prefill_cuda_graph.integration == "incompatible"


@pytest.mark.parametrize("integration", ["direct", "adapter"])
def test_model_capabilities_accept_prefill_cuda_graph_integration(
    integration: str,
) -> None:
    backend = PrefillCudaGraphBackendCapability(
        backend="breakable",
        status="validated",
        default_token_buckets=(32, 64),
    )

    capabilities = _capabilities_with_prefill(
        PrefillCudaGraphCapability(
            integration=integration,
            backends=(backend,),
            preferred_backend="breakable",
        )
    )

    assert capabilities.prefill_cuda_graph is not None
    assert capabilities.prefill_cuda_graph.integration == integration


@pytest.mark.parametrize("integration", ["direct", "adapter"])
def test_prefill_cuda_graph_integration_does_not_require_backends(
    integration: str,
) -> None:
    capabilities = _capabilities_with_prefill(
        PrefillCudaGraphCapability(integration=integration)
    )

    assert capabilities.prefill_cuda_graph is not None
    assert capabilities.prefill_cuda_graph.backends == ()


def test_model_capabilities_accept_incompatible_prefill_cuda_graph() -> None:
    capabilities = _capabilities_with_prefill(
        PrefillCudaGraphCapability(
            integration="incompatible",
            incompatible_reason="model forward is not graph compatible",
        )
    )

    assert capabilities.prefill_cuda_graph is not None
    assert capabilities.prefill_cuda_graph.incompatible_reason is not None


def test_model_capabilities_reject_incompatible_without_reason() -> None:
    with pytest.raises(ValueError, match="requires incompatible_reason"):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(integration="incompatible")
        )


def test_model_capabilities_reject_duplicate_prefill_backends() -> None:
    backend = PrefillCudaGraphBackendCapability(
        backend="breakable",
        status="validated",
        default_token_buckets=(32,),
    )

    with pytest.raises(ValueError, match="duplicate.*backend"):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(
                integration="direct",
                backends=(backend, backend),
            )
        )


def test_model_capabilities_reject_invalid_preferred_backend() -> None:
    with pytest.raises(ValueError, match="preferred.*present"):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(
                integration="adapter",
                preferred_backend="breakable",
            )
        )


def test_model_capabilities_reject_blocked_backend_without_reason() -> None:
    with pytest.raises(ValueError, match="blocked.*requires a reason"):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(
                integration="direct",
                backends=(
                    PrefillCudaGraphBackendCapability(
                        backend="full",
                        status="blocked",
                    ),
                ),
            )
        )


def test_model_capabilities_reject_incompatible_backend_declarations() -> None:
    with pytest.raises(ValueError, match="incompatible.*cannot declare backends"):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(
                integration="incompatible",
                incompatible_reason="unsupported forward",
                backends=(
                    PrefillCudaGraphBackendCapability(
                        backend="full",
                        status="validated",
                        default_token_buckets=(32,),
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    ("buckets", "error"),
    [
        ((), "must not be empty"),
        ((0,), "positive"),
        ((32, 16), "strictly increasing"),
        ((32, 32), "unique"),
        ((32, 64.0), "integers"),
    ],
)
def test_model_capabilities_reject_invalid_default_token_buckets(
    buckets: tuple[int, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _capabilities_with_prefill(
            PrefillCudaGraphCapability(
                integration="direct",
                backends=(
                    PrefillCudaGraphBackendCapability(
                        backend="breakable",
                        status="validated",
                        default_token_buckets=buckets,
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    ("status", "reason"),
    [("blocked", "not supported yet"), ("unvalidated", None)],
)
def test_non_enableable_prefill_backend_allows_empty_default_token_buckets(
    status: str,
    reason: str | None,
) -> None:
    capabilities = _capabilities_with_prefill(
        PrefillCudaGraphCapability(
            integration="direct",
            backends=(
                PrefillCudaGraphBackendCapability(
                    backend="full",
                    status=status,
                    reason=reason,
                ),
            ),
        )
    )

    assert capabilities.prefill_cuda_graph is not None
    assert capabilities.prefill_cuda_graph.backends[0].default_token_buckets == ()


@pytest.mark.parametrize("architecture", EXPECTED_TTS_CAPABILITIES)
def test_tts_model_package_exports_capabilities(architecture: str) -> None:
    module = _package_for_architecture(architecture)
    capabilities = getattr(module, "CAPABILITIES", None)

    assert capabilities == EXPECTED_TTS_CAPABILITIES[architecture]
    assert isinstance(capabilities, ModelCapabilities)
    for field in fields(ModelCapabilities)[:-1]:
        assert isinstance(getattr(capabilities, field.name), bool)
    assert (
        capabilities.prefill_cuda_graph
        == EXPECTED_TTS_CAPABILITIES[architecture].prefill_cuda_graph
    )


@pytest.mark.parametrize("architecture", EXPECTED_TTS_CAPABILITIES)
def test_get_model_capabilities_for_tts_architecture(architecture: str) -> None:
    assert (
        get_model_capabilities(architecture) == EXPECTED_TTS_CAPABILITIES[architecture]
    )


def test_get_model_capabilities_for_non_tts_and_unknown_architectures() -> None:
    assert get_model_capabilities("Qwen3OmniMoeForConditionalGeneration") is None
    assert get_model_capabilities("UnknownArchitecture") is None


def test_get_model_capabilities_rejects_malformed_capabilities_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "tests.unit_test.models.fake_bad_capabilities"
    fake_package = ModuleType(package_name)
    fake_package.CAPABILITIES = object()

    class FakeConfig:
        pass

    FakeConfig.__module__ = f"{package_name}.config"

    monkeypatch.setitem(sys.modules, package_name, fake_package)
    monkeypatch.setitem(
        PIPELINE_CONFIG_REGISTRY.configs,
        "MalformedCapabilitiesModel",
        FakeConfig,
    )

    with pytest.raises(TypeError, match="must be a ModelCapabilities instance"):
        get_model_capabilities("MalformedCapabilitiesModel")


def test_get_model_capabilities_resolves_registered_alias() -> None:
    assert (
        get_model_capabilities("MossTTSDelay")
        == EXPECTED_TTS_CAPABILITIES["MossTTSDelayModel"]
    )


def test_model_capabilities_are_static_architecture_metadata() -> None:
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
    custom_config = config_cls(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    capabilities = get_model_capabilities(config_cls.architecture)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is True
    assert custom_config.supports_uploaded_voice_references() is False


def test_launcher_model_capabilities_log_summary() -> None:
    from sglang_omni.serve.launcher import _model_capabilities_log_summary

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
    summary = _model_capabilities_log_summary(
        config_cls(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    )

    assert summary == {
        "architecture": "Qwen3TTSForConditionalGeneration",
        "reference_audio": True,
        "batch_vocoder": True,
        "streaming_vocoder": True,
        "cuda_graph": True,
        "torch_compile": True,
    }


def test_launcher_model_capabilities_log_summary_uses_static_architecture() -> None:
    from sglang_omni.serve.launcher import _model_capabilities_log_summary

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")
    summary = _model_capabilities_log_summary(
        config_cls(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    )

    assert summary is not None
    assert summary["reference_audio"] is True
    assert summary["batch_vocoder"] is True


def test_launcher_emits_model_capabilities_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sglang_omni.serve.launcher import _log_model_capabilities

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(
        "VoxtralTTSForConditionalGeneration"
    )
    with caplog.at_level("INFO", logger="sglang_omni.serve.launcher"):
        _log_model_capabilities(config_cls(model_path="dummy"))

    assert "Model capabilities:" in caplog.text
    assert '"architecture": "VoxtralTTSForConditionalGeneration"' in caplog.text
    assert '"reference_audio": false' in caplog.text
    assert '"batch_vocoder": false' in caplog.text


def test_launcher_model_capabilities_warning_isolated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sglang_omni.serve import launcher

    def fail_summary(_pipeline_config: object) -> None:
        raise RuntimeError("capability lookup failed")

    monkeypatch.setattr(launcher, "_model_capabilities_log_summary", fail_summary)
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")

    with caplog.at_level("WARNING", logger="sglang_omni.serve.launcher"):
        launcher._log_model_capabilities(config_cls(model_path="dummy"))

    assert "Failed to resolve model capabilities for startup log" in caplog.text
