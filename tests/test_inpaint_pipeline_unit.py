# tests/test_inpaint_pipeline_unit.py
import torch
import pytest


def test_inpaint_mask_unit_outputs_mask_bool_when_local_enabled():
    from diffsynth.pipelines.wan_video import WanVideoUnit_InpaintMask

    unit = WanVideoUnit_InpaintMask()

    # Fake pipe (only attributes the unit reads)
    class FakePipe: torch_dtype = torch.float32; device = torch.device("cpu")
    pipe = FakePipe()

    # vace_video_mask: (B=1, 1, T=5, H=16, W=16)
    vace_video_mask = torch.zeros(1, 1, 5, 16, 16)
    vace_video_mask[0, 0, 0, :2, :2] = 1.0      # hot top-left of frame 0

    # latents: (B=1, 16, T_lat=2, H/8=2, W/8=2) -- already VAE-encoded
    latents = torch.zeros(1, 16, 2, 2, 2)

    out = unit.process(
        pipe,
        vace_video_mask=vace_video_mask,
        latents=latents,
        inpaint_local_enabled=True,
        inpaint_global_enabled=False,
    )
    assert "mask_bool_latent" in out
    assert "mask_bool_token" in out
    # mask_bool_latent at factor=8: (T_lat=2, H/8=2, W/8=2) flattened -> 8
    assert out["mask_bool_latent"].shape == (8,)
    assert out["mask_bool_latent"].dtype == torch.bool
    # Token grid: VAE latent then patch_size (1,2,2) -> (2, 1, 1) -> 2
    assert out["mask_bool_token"].shape == (2,)
    # downsampled_input_latents only present when global is enabled
    assert out.get("downsampled_input_latents") is None


def test_inpaint_mask_unit_outputs_downsampled_latents_when_global_enabled():
    from diffsynth.pipelines.wan_video import WanVideoUnit_InpaintMask

    unit = WanVideoUnit_InpaintMask()

    class FakePipe: torch_dtype = torch.float32; device = torch.device("cpu")
    pipe = FakePipe()

    vace_video_mask = torch.zeros(1, 1, 5, 16, 16)
    latents = torch.randn(1, 16, 2, 4, 4)

    out = unit.process(
        pipe,
        vace_video_mask=vace_video_mask,
        latents=latents,
        inpaint_local_enabled=True,
        inpaint_global_enabled=True,
    )
    assert out.get("downsampled_input_latents") is not None
    # area-interp by 0.5 on spatial dims only
    assert out["downsampled_input_latents"].shape[-2:] == (2, 2)
    assert out["downsampled_input_latents"].shape[2] == 2  # temporal preserved


def test_inpaint_mask_unit_noop_when_flags_off():
    from diffsynth.pipelines.wan_video import WanVideoUnit_InpaintMask

    unit = WanVideoUnit_InpaintMask()

    class FakePipe: torch_dtype = torch.float32; device = torch.device("cpu")
    pipe = FakePipe()

    out = unit.process(
        pipe,
        vace_video_mask=None,
        latents=torch.zeros(1, 16, 2, 4, 4),
        inpaint_local_enabled=False,
        inpaint_global_enabled=False,
    )
    # All three outputs must be None when flags are off
    assert out.get("mask_bool_latent") is None
    assert out.get("mask_bool_token") is None
    assert out.get("downsampled_input_latents") is None
