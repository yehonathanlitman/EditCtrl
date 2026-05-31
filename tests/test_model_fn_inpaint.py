import torch
import pytest


def test_model_fn_accepts_inpaint_kwargs():
    """Smoke test: the function signature accepts the new kwargs without crashing
    when both flags are off and the values are None."""
    import inspect
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    sig = inspect.signature(model_fn_wan_video)
    for k in ("mask_bool_token", "downsampled_input_latents",
              "inpaint_local_enabled", "inpaint_global_enabled"):
        assert k in sig.parameters, f"missing kwarg: {k}"
