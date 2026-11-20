import pytest
import torch
from torch import nn

from rlllmf.config import validate_settings
from rlllmf.offload import ReferenceOffloader
from tests.test_core import temporary_settings


def reference_model(device="cpu"):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])
    return model.to(device).requires_grad_(False)


def test_budget_selects_only_fitting_layers_and_preserves_weights_on_exception():
    model = reference_model()
    before = {name: p.clone() for name, p in model.named_parameters()}
    controller = ReferenceOffloader(model, 128 / 1024**3, "cpu")
    assert controller.weight_bytes == 128
    assert len(controller.layers) == 2
    with pytest.raises(RuntimeError, match="simulated failure"):
        with controller.on_execution_device():
            raise RuntimeError("simulated failure")
    assert controller.active is False
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    assert all(p.device.type == "cpu" for _, layer in controller.layers for p in layer.parameters())


def test_trainable_model_cannot_be_offloaded_and_tiny_budget_is_rejected():
    model = reference_model()
    with pytest.raises(ValueError, match="too small"):
        ReferenceOffloader(model, 1 / 1024**3, "cpu")
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen reference"):
        ReferenceOffloader(model, 1, "cpu")


def test_native_lfm_checkpoint_resume_restores_all_converted_parameter_names(tmp_path):
    from transformers import Lfm3Config, Lfm3ForCausalLM

    from rlllmf.offload import OffloadGRPOTrainer

    config = Lfm3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, conv_L_cache=7,
        layer_types=["conv", "full_attention"], bos_token_id=None,
        pad_token_id=0, eos_token_id=1,
    )
    source = Lfm3ForCausalLM(config)
    expected = {name: value.detach().clone() for name, value in source.state_dict().items()}
    source.save_pretrained(tmp_path)
    trainer = OffloadGRPOTrainer.__new__(OffloadGRPOTrainer)
    trainer.model = Lfm3ForCausalLM(config)
    trainer._load_from_checkpoint(str(tmp_path))
    for name, actual in trainer.model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("budget", [-1, float("inf"), float("nan"), True, "2"])
def test_invalid_offload_budget_is_rejected(tmp_path, budget):
    settings = temporary_settings(tmp_path)
    settings.training.reference_cpu_offload_gib = budget
    with pytest.raises(ValueError, match="reference_cpu_offload_gib"):
        validate_settings(settings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for residency checks")
def test_partial_gpu_offload_preserves_output_and_returns_selected_layers_to_cpu():
    model = reference_model("cuda")
    inputs = torch.randn(2, 4, device="cuda")

    def forward():
        value = inputs
        for layer in model.model.layers:
            value = layer(value)
        return value

    with torch.no_grad():
        expected = forward()
        controller = ReferenceOffloader(model, 128 / 1024**3, "cuda")
        assert next(model.model.layers[0].parameters()).device.type == "cuda"
        assert next(model.model.layers[1].parameters()).device.type == "cpu"
        with controller.on_execution_device():
            assert all(p.device.type == "cuda" for p in model.parameters())
            torch.testing.assert_close(forward(), expected)
        assert next(model.model.layers[0].parameters()).device.type == "cuda"
        assert next(model.model.layers[1].parameters()).device.type == "cpu"
        assert controller.cycles == 1
