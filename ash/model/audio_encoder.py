"""Audio encoder: pretrained Whisper v3 encoder with MLP projection into LLM space."""
from __future__ import annotations

import torch
import torch.nn as nn

from ash.config import AudioConfig


class AudioProjector(nn.Module):
    """2-layer MLP that maps audio encoder outputs to the LLM embedding dimension."""

    def __init__(self, encoder_dim: int, projection_dim: int, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = encoder_dim
        for i in range(n_layers):
            out_dim = projection_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_frames, encoder_dim] -> [B, N_frames, projection_dim]"""
        return self.mlp(x)


class AudioEncoder(nn.Module):
    """
    Wraps the encoder portion of a pretrained Whisper model and projects
    its outputs into the LLM embedding space via a learned MLP projector.

    Only the encoder is used — the Whisper decoder is not loaded.
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        # Loaded lazily or explicitly via load_pretrained()
        self.encoder: nn.Module | None = None
        self.projector = AudioProjector(
            config.encoder_hidden_dim,
            config.projection_dim,
            config.projection_layers,
        )

    def load_pretrained(self) -> None:
        """Load the pretrained Whisper encoder from HuggingFace."""
        from transformers import WhisperModel

        whisper = WhisperModel.from_pretrained(self.config.encoder_name)
        self.encoder = whisper.encoder
        # Discard decoder to save memory
        del whisper

        if self.config.freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (projector remains trainable)."""
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for finetuning."""
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = True

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of audio spectrograms.

        Args:
            input_features: [B, n_mels, T_mel] Whisper-style mel spectrogram

        Returns:
            [B, T_enc, projection_dim] projected audio tokens
        """
        if self.encoder is None:
            self.load_pretrained()

        outputs = self.encoder(input_features=input_features)
        hidden = outputs.last_hidden_state  # [B, T_enc, encoder_dim]
        return self.projector(hidden)
