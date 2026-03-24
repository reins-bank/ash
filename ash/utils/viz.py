from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_esc_heatmap(
    esc_scores_per_layer: list[torch.Tensor],
    tokens: list[str] | None = None,
) -> Figure:
    """
    Plot ESC scores as a [n_layers x seq_len] heatmap.

    Args:
        esc_scores_per_layer: list of [B, T, 1] tensors (uses first batch element)
        tokens: optional token labels for x-axis
    """
    import matplotlib.pyplot as plt

    # Stack: [n_layers, T]
    data = torch.stack([s[0, :, 0] for s in esc_scores_per_layer]).detach().cpu().numpy()
    n_layers, seq_len = data.shape

    fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.15), 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("ESC Scores (0=open, 1=closed)")
    plt.colorbar(im, ax=ax)

    if tokens and len(tokens) <= 50:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)

    fig.tight_layout()
    return fig


def plot_suppression_mask(
    suppress_mask: torch.Tensor,
    tokens: list[str] | None = None,
    layer: int = 0,
) -> Figure:
    """
    Plot [T x T] ASH suppression pattern for a single layer.

    Args:
        suppress_mask: [B, n_ash, T, T] tensor (uses first batch, mean across ASH heads)
        tokens: optional token labels
    """
    import matplotlib.pyplot as plt

    data = suppress_mask[0].mean(dim=0).detach().cpu().numpy()  # [T, T]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"ASH Suppression Mask (Layer {layer})")
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig
