# diffsynth/utils/inpaint_mask.py
"""Mask helpers for VACE inpainting (Wan 2.1).

Ported from DiffSynthInpaint/diffsynth/pipelines/wan_video_new.py:30-106.
"""
import torch
import torch.nn.functional as F


def downsample_tensor(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Max-pool a 2D tensor by `factor` on each spatial dim."""
    t = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    pooled = F.max_pool2d(t, kernel_size=factor, stride=factor)
    return pooled.squeeze(0).squeeze(0)


def compute_downsampled_masks_tensor(masks: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Downsample a video mask onto the Wan VAE latent grid.

    Args:
        masks: (C, N, H, W). H and W must be divisible by `factor`.
        factor: spatial max-pool factor. 8 -> latent grid, 16 -> DiT patch grid.

    Returns:
        (C, M, H/factor, W/factor) where M = 1 + (N-1)//4. Frame 0 is downsampled
        directly; each subsequent group of <=4 frames is max-unioned then
        downsampled. This mirrors Wan VAE temporal compression (T -> 1 + (T-1)/4).
    """
    C, N, H, W = masks.shape
    assert H % factor == 0 and W % factor == 0, "H and W must be divisible by factor"

    outputs = []
    for c in range(C):
        ch = masks[c]
        per_channel = [downsample_tensor(ch[0], factor)]
        for i in range(1, N, 4):
            chunk = ch[i : i + 4]
            union = torch.max(chunk, dim=0).values
            per_channel.append(downsample_tensor(union, factor))
        outputs.append(torch.stack(per_channel, dim=0))
    return torch.stack(outputs, dim=0)


def dilate_mask(mask: torch.Tensor, dilation_radius: int) -> torch.Tensor:
    """Square-kernel binary dilation. mask shape: (B, T, H, W)."""
    B, T, H, W = mask.shape
    if dilation_radius <= 0:
        return mask
    ksize = 2 * dilation_radius + 1
    kernel = torch.ones((1, 1, ksize, ksize), device=mask.device, dtype=torch.float32)
    flat = mask.view(B * T, 1, H, W).float()
    dilated = F.conv2d(flat, kernel, padding=dilation_radius)
    return (dilated > 0).to(mask.dtype).view(B, T, H, W)
