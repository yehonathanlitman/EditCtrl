# tests/test_inpaint_mask_utils.py
import torch
import pytest
from diffsynth.utils.inpaint_mask import (
    downsample_tensor,
    compute_downsampled_masks_tensor,
    dilate_mask,
)


def test_downsample_tensor_max_pool_factor_2():
    # 4x4 tensor with a single hot pixel at (0,0); factor=2 -> 2x2 output, top-left only
    t = torch.zeros(4, 4)
    t[0, 0] = 1.0
    out = downsample_tensor(t, factor=2)
    assert out.shape == (2, 2)
    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 0.0
    assert out[1, 0].item() == 0.0


def test_compute_downsampled_masks_tensor_shape():
    # mask: (C=1, N=5, H=16, W=16), factor=8
    # Output M = 1 + (5-1)//4 = 2 frames, spatial 16/8 = 2.
    mask = torch.zeros(1, 5, 16, 16)
    out = compute_downsampled_masks_tensor(mask, factor=8)
    assert out.shape == (1, 2, 2, 2)


def test_compute_downsampled_masks_tensor_max_union_over_chunks_of_4():
    # First frame keeps its mask; frames 1..4 are max-unioned then pooled.
    mask = torch.zeros(1, 5, 16, 16)
    mask[0, 0, 0, 0] = 1.0       # first frame hot top-left
    mask[0, 3, 8, 8] = 1.0       # frame 3 hot middle (within the 1..4 union chunk)
    out = compute_downsampled_masks_tensor(mask, factor=8)
    # Output frame 0 reflects original frame 0
    assert out[0, 0, 0, 0].item() == 1.0
    # Output frame 1 reflects union of input frames 1..4, then 8x8 max-pool -> hot at (1, 1)
    assert out[0, 1, 1, 1].item() == 1.0


def test_compute_downsampled_masks_tensor_rejects_non_divisible():
    mask = torch.zeros(1, 5, 15, 16)
    with pytest.raises(AssertionError):
        compute_downsampled_masks_tensor(mask, factor=8)


def test_dilate_mask_radius_1_grows_one_pixel_ring():
    mask = torch.zeros(1, 1, 5, 5)
    mask[0, 0, 2, 2] = 1.0
    out = dilate_mask(mask, dilation_radius=1)
    # Original center plus 8 neighbours
    assert out[0, 0, 1:4, 1:4].sum().item() == 9.0
    # Corners untouched
    assert out[0, 0, 0, 0].item() == 0.0
