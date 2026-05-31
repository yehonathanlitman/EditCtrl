import torch
import pytest

from diffsynth.models.wan_video_vace import VaceWanModel


def _tiny_model():
    return VaceWanModel(
        vace_layers=(0,),
        vace_in_dim=8,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=16,
        num_heads=2,
        ffn_dim=32,
    )


def _make_inputs(seq=12, dim=16, num_heads=2):
    """Build valid minimal inputs for a tiny VaceWanModel.

    Key shapes:
      x         : (B, seq, dim)
      context   : (B, ctx_len, dim)  — cross-attn context
      t_mod     : (B, 6, dim)        — adaLN params (6 per block)
      freqs     : (seq, 1, head_dim//2) complex64  — RoPE freqs

    freqs must be complex because rope_apply calls view_as_complex on the
    query/key tensors and multiplies element-wise by freqs.
    head_dim = dim // num_heads = 8, so head_dim//2 = 4.
    """
    B = 1
    head_dim = dim // num_heads
    # vace_context: Conv3d(vace_in_dim=8, dim, patch=(1,2,2))
    # Input shape (8, F, H, W); with F=1, H=4, W=6 -> output grid (1, 2, 3) = 6 tokens
    vace_context = [torch.zeros(8, 1, 4, 6)]
    x = torch.zeros(B, seq, dim)
    context = torch.zeros(B, 1, dim)
    t_mod = torch.zeros(B, 6, dim)
    # freqs: complex tensor of shape (seq, 1, head_dim//2)
    freqs = torch.zeros(seq, 1, head_dim // 2, dtype=torch.complex64)
    return x, vace_context, context, t_mod, freqs


def test_vace_forward_mask_bool_none_matches_baseline_shape():
    m = _tiny_model().eval()
    B, seq, dim = 1, 12, 16
    x, vace_context, context, t_mod, freqs = _make_inputs(seq=seq, dim=dim)

    hints = m(x, vace_context, context, t_mod, freqs)
    # Returns a tuple of length len(vace_layers); each hint matches x seq length
    assert len(hints) == 1
    assert hints[0].shape == (B, seq, dim)


def test_vace_forward_mask_bool_scatters_zeros_outside_mask():
    """When mask_bool selects a subset of token positions, the returned hint
    must have non-zero values only at those positions (after scatter-back)."""
    torch.manual_seed(0)
    m = _tiny_model().eval()
    # Initialise after_proj with a known scale so output is detectable
    for block in m.vace_blocks:
        torch.nn.init.constant_(block.after_proj.weight, 0.1)
        torch.nn.init.constant_(block.after_proj.bias, 0.0)

    B, seq, dim = 1, 12, 16
    x_rand = torch.randn(B, seq, dim)
    vace_context = [torch.randn(8, 1, 4, 6)]
    context = torch.zeros(B, 1, dim)
    t_mod = torch.zeros(B, 6, dim)
    _, _, _, _, freqs = _make_inputs(seq=seq, dim=dim)

    mask_bool = torch.zeros(seq, dtype=torch.bool)
    mask_bool[2:6] = True  # 4 active token positions

    hints = m(x_rand, vace_context, context, t_mod, freqs, mask_bool=mask_bool)
    assert hints[0].shape == (B, seq, dim)
    # Positions outside mask must be zero
    outside = hints[0][:, ~mask_bool, :]
    assert torch.all(outside == 0), "hint values outside mask must be zero"
    # Positions inside mask must be non-zero (with positive-init weights and random x)
    inside = hints[0][:, mask_bool, :]
    assert torch.any(inside != 0), "hint values inside mask must be non-zero"
