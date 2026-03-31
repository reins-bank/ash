"""Multimodal embedding layer: replaces special tokens with encoder outputs."""
from __future__ import annotations

import torch
import torch.nn as nn

from ash.config import ModelConfig
from ash.model.embeddings import Embeddings
from ash.model.vision_encoder import VisionEncoder
from ash.model.audio_encoder import AudioEncoder


class MultimodalEmbeddings(nn.Module):
    """
    Extends the base text embeddings to support multimodal inputs.

    Special tokens (<image>, <video>, <audio>) in the input sequence are
    replaced with projected encoder outputs from the vision/audio encoders.
    Text tokens are embedded normally via the base Embeddings module.
    """

    def __init__(self, config: ModelConfig, base_embeddings: Embeddings):
        super().__init__()
        self.config = config
        self.base_embeddings = base_embeddings
        mm = config.multimodal

        self.vision_encoder: VisionEncoder | None = None
        self.audio_encoder: AudioEncoder | None = None

        if mm.vision.enabled:
            self.vision_encoder = VisionEncoder(mm.vision)
        if mm.audio.enabled:
            self.audio_encoder = AudioEncoder(mm.audio)

    def forward(
        self,
        idx: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Build the full embedding sequence, replacing special tokens with
        projected encoder outputs.

        Args:
            idx: [B, T] input token ids (with special placeholder tokens)
            images: [B, C, H, W] or None — single image per sample
            video_frames: [B, N_frames, C, H, W] or None
            audio_features: [B, n_mels, T_mel] or None
            image_token_id: vocab id for <image> placeholder
            video_token_id: vocab id for <video> placeholder
            audio_token_id: vocab id for <audio> placeholder

        Returns:
            [B, T_expanded, d_model] embedding tensor with encoder outputs
            replacing the special token positions.
        """
        # Start with base text embeddings
        embeddings = self.base_embeddings(idx)  # [B, T, d_model]

        # Encode and inject vision tokens (image)
        if images is not None and self.vision_encoder is not None and image_token_id is not None:
            vision_tokens = self.vision_encoder(pixel_values=images)  # [B, N_patches, d_model]
            embeddings = self._replace_tokens(
                embeddings, idx, image_token_id, vision_tokens
            )

        # Encode and inject vision tokens (video)
        if video_frames is not None and self.vision_encoder is not None and video_token_id is not None:
            video_tokens = self.vision_encoder(frames=video_frames)  # [B, N_total, d_model]
            embeddings = self._replace_tokens(
                embeddings, idx, video_token_id, video_tokens
            )

        # Encode and inject audio tokens
        if audio_features is not None and self.audio_encoder is not None and audio_token_id is not None:
            audio_tokens = self.audio_encoder(audio_features)  # [B, T_enc, d_model]
            embeddings = self._replace_tokens(
                embeddings, idx, audio_token_id, audio_tokens
            )

        return embeddings

    @staticmethod
    def _replace_tokens(
        embeddings: torch.Tensor,
        idx: torch.Tensor,
        token_id: int,
        replacement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace positions where idx == token_id with encoder output tokens.

        For simplicity, assumes the special token appears once per sample
        and the replacement tokens expand in place. In practice, the
        data collator pre-pads the sequence to account for the expanded
        length of encoder outputs.

        This is a simple first-pass implementation. Production multimodal
        models use more sophisticated interleaving with per-sample offsets.
        """
        B = idx.shape[0]
        for b in range(B):
            mask = idx[b] == token_id
            positions = mask.nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue
            # Replace the first occurrence with encoder tokens
            start = positions[0].item()
            n_tokens = min(replacement.shape[1], embeddings.shape[1] - start)
            embeddings[b, start : start + n_tokens] = replacement[b, :n_tokens]
        return embeddings
