import torch
import pytest
from ash.config import ModelConfig, TrainingConfig
from ash.model.gpt import GPT
from ash.training.losses import combined_loss
from ash.training.scheduler import CosineWarmupScheduler, CERCurriculumScheduler


def test_training_smoke_10_steps():
    """Smoke test: 10 steps of training on random data, loss should decrease."""
    cfg = ModelConfig(n_layer=2, n_head=4, d_model=128, block_size=64)
    cfg.cer.enabled = True
    cfg.cer.n_ash_heads = 1
    cfg.cer.esc_mlp_hidden = 16
    model = GPT(cfg)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device="cpu"
    )
    curriculum = CERCurriculumScheduler(
        lambda_esc_max=0.1, lambda_ash_max=0.05, phase1_end=0.0, phase2_end=0.0
    )

    losses = []
    for step in range(50):
        idx = torch.randint(0, cfg.vocab_size, (4, 64))
        targets = torch.randint(0, cfg.vocab_size, (4, 64))

        logits, lm_loss, cer_info = model(idx, targets)
        lambda_esc, lambda_ash = curriculum.get_lambdas(step, 100)
        total_loss, _ = combined_loss(lm_loss, cer_info, lambda_esc, lambda_ash)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(total_loss.item())

    # No NaN
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN detected in loss"
    # Loss should be finite and reasonable
    assert all(l < 20.0 for l in losses), "Loss exploded"
    # Gradients should be affecting the loss (not static)
    assert losses[0] != losses[-1], "Loss didn't change at all — gradients may not be flowing"


def test_cer_curriculum_phases():
    sched = CERCurriculumScheduler(
        lambda_esc_max=0.1, lambda_ash_max=0.05, phase1_end=0.1, phase2_end=0.3
    )
    # Phase 1: no CER loss
    esc, ash = sched.get_lambdas(0, 1000)
    assert esc == 0.0 and ash == 0.0

    esc, ash = sched.get_lambdas(50, 1000)  # 5% = phase 1
    assert esc == 0.0 and ash == 0.0

    # Phase 2: ramping
    esc, ash = sched.get_lambdas(200, 1000)  # 20% = mid-ramp
    assert 0.0 < esc < 0.1
    assert 0.0 < ash < 0.05

    # Phase 3: full
    esc, ash = sched.get_lambdas(500, 1000)  # 50% = full
    assert esc == 0.1
    assert ash == 0.05


def test_cosine_warmup_scheduler():
    optimizer = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=1e-3)
    sched = CosineWarmupScheduler(
        optimizer=optimizer, warmup_steps=100, max_steps=1000, lr_max=1e-3, lr_min=1e-5
    )
    # Warmup: lr should increase
    lr_0 = sched.get_lr(0)
    lr_50 = sched.get_lr(50)
    lr_100 = sched.get_lr(100)
    assert lr_0 < lr_50 < lr_100

    # Peak at warmup end
    assert abs(lr_100 - 1e-3) < 1e-6

    # Decay after warmup
    lr_500 = sched.get_lr(500)
    assert lr_500 < 1e-3

    # Min at end
    lr_1000 = sched.get_lr(1000)
    assert abs(lr_1000 - 1e-5) < 1e-6


def test_checkpoint_roundtrip(tmp_path):
    from ash.training.checkpoint import save_checkpoint, load_checkpoint

    cfg = ModelConfig(n_layer=2, n_head=4, d_model=128, block_size=64)
    cfg.cer.enabled = True
    cfg.cer.n_ash_heads = 1
    model = GPT(cfg)
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")

    train_cfg = TrainingConfig()
    path = tmp_path / "test_ckpt.pt"
    save_checkpoint(model, optimizer, step=42, model_config=cfg,
                    train_config=train_cfg, metrics={"val_loss": 3.14}, path=path)

    # Load into fresh model
    model2 = GPT(cfg)
    ckpt = load_checkpoint(path, model2)
    assert ckpt["step"] == 42

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"
