import torch
import pytest
from ash.config import ModelConfig
from ash.model.gpt import GPT


def test_vanilla_gpt2_param_count():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    model = GPT(cfg)
    counts = model.param_count()
    # GPT-2 124M (standard)
    assert 120_000_000 < counts["total"] < 130_000_000
    assert counts["cer"] == 0


def test_cer_model_param_count():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    model = GPT(cfg)
    counts = model.param_count()
    # Base + CER overhead
    assert counts["cer"] > 0
    assert counts["total"] == counts["base"] + counts["cer"]
    # CER should be ~5-7% of base
    overhead = counts["cer"] / counts["base"]
    assert 0.04 < overhead < 0.10


def test_forward_pass_with_loss():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss, cer_info = model(idx, targets)
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_forward_pass_no_targets():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    logits, loss, cer_info = model(idx)
    assert logits.shape == (1, 32, cfg.vocab_size)
    assert loss is None


def test_cer_info_populated():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    _, _, cer_info = model(idx)
    assert "esc_scores_per_layer" in cer_info
    assert len(cer_info["esc_scores_per_layer"]) == cfg.n_layer


def test_cer_disabled_no_info():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    _, _, cer_info = model(idx)
    assert cer_info == {}


def test_generate():
    cfg = ModelConfig()
    cfg.cer.enabled = False
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)  # 4 input + 10 generated


def test_generate_with_cer():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)


def test_backward_pass():
    cfg = ModelConfig()
    cfg.cer.enabled = True
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 32))
    targets = torch.randint(0, cfg.vocab_size, (1, 32))
    _, loss, _ = model(idx, targets)
    loss.backward()
    # Check all params got gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
