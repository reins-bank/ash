"""Vision encoder: pretrained SigLIP 2 with MLP projection into LLM space."""
from __future__ import annotations

import torch
import torch.nn as nn

from ash.config import VisionConfig


class VisionProjector(nn.Module):
    """2-layer MLP that maps vision encoder outputs to the LLM embedding dimension."""

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
        """x: [B, N_patches, encoder_dim] -> [B, N_patches, projection_dim]"""
        return self.mlp(x)


class VisionEncoder(nn.Module):
    """
    Wraps a pretrained vision encoder (SigLIP 2 by default) and projects
    its outputs into the LLM embedding space via a learned MLP projector.

    Image inputs are encoded into patch tokens; video inputs are treated
    as a sequence of frames, each independently encoded, then concatenated.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        # The pretrained encoder is loaded lazily at first forward pass
        # or explicitly via load_pretrained(). This avoids importing
        # transformers at module definition time.
        self.encoder: nn.Module | None = None
        self.projector = VisionProjector(
            config.encoder_hidden_dim,
            config.projection_dim,
            config.projection_layers,
        )

    def load_pretrained(self) -> None:
        """Load the pretrained vision encoder from HuggingFace."""
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            self.config.encoder_name,
            trust_remote_code=True,
        )
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

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images.

        Args:
            pixel_values: [B, C, H, W] preprocessed image tensor

        Returns:
            [B, N_patches, projection_dim] projected vision tokens
        """
        if self.encoder is None:
            self.load_pretrained()

        # SigLIP / ViT models return last_hidden_state as patch embeddings
        outputs = self.encoder(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [B, N_patches, encoder_dim]
        return self.projector(hidden)

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of videos (as sampled frames).

        Args:
            frames: [B, N_frames, C, H, W] preprocessed video frames

        Returns:
            [B, N_frames * N_patches, projection_dim] projected vision tokens
        """
        B, N_frames, C, H, W = frames.shape
        # Flatten batch and frames, encode all at once
        flat = frames.view(B * N_frames, C, H, W)
        tokens = self.encode_image(flat)  # [B*N_frames, N_patches, dim]
        N_patches = tokens.shape[1]
        # Reshape back: all frame tokens concatenated per sample
        return tokens.view(B, N_frames * N_patches, -1)

    def forward(
        self, pixel_values: torch.Tensor | None = None, frames: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Convenience forward: dispatches to encode_image or encode_video."""
        if frames is not None:
            return self.encode_video(frames)
        if pixel_values is not None:
            return self.encode_image(pixel_values)
        raise ValueError("Either pixel_values or frames must be provided")
