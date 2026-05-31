import torch
import pytest

from diffsynth.diffusion.loss import masked_token_mse, WanVideoInpaintMaskedLoss


def test_masked_token_mse_zero_when_pred_matches_target_inside_mask():
    B, C, T, H, W = 1, 16, 2, 4, 4
    pred = torch.randn(B, C, T, H, W)
    target = pred.clone()
    target[0, 0, 0, 0, 0] += 5.0   # error OUTSIDE the mask
    mask = torch.zeros(T * H * W, dtype=torch.bool)
    mask[1:5] = True               # 4 active token positions
    loss = masked_token_mse(pred, target, mask)
    # Error is outside mask -> loss should be 0
    assert torch.isclose(loss, torch.tensor(0.0))


def test_masked_token_mse_picks_up_inside_mask_error():
    B, C, T, H, W = 1, 16, 2, 4, 4
    pred = torch.zeros(B, C, T, H, W)
    target = torch.zeros(B, C, T, H, W)
    # Place a delta at token index 1, channel 0
    target.view(B, C, -1)[0, 0, 1] = 1.0
    mask = torch.zeros(T * H * W, dtype=torch.bool)
    mask[1] = True
    loss = masked_token_mse(pred, target, mask)
    # MSE over the masked-token positions: 1 channel-0 position has error^2 = 1.0,
    # remaining 15 channels at the same spatial token have error 0. mean = 1/16 = 0.0625
    assert torch.isclose(loss, torch.tensor(0.0625))


def test_wan_video_inpaint_masked_loss_importable():
    assert WanVideoInpaintMaskedLoss is not None
