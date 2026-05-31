# Wan 2.1 VACE Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the two-stage token-level latent-mask inpainting mechanism from `DiffSynthInpaint` into fresh `diffsynth-studio/` for Wan 2.1 VACE (1.3B + 14B), as small flag-gated extensions to existing files.

**Architecture:** All inpainting logic is gated by three flags (`enable_inpaint_local`, `enable_inpaint_global`, `stage2_freeze`) + one new task value (`sft:inpaint`). Default upstream behavior is bit-identical when flags are off. VACE forward gains a `mask_bool` arg that slices the control sequence to mask tokens; `model_fn_wan_video` gains a global path that concatenates downsampled-input-latent tokens into the cross-attention context; `WanVideoPipeline.__call__` gains optional outside-mask latent surgery.

**Tech Stack:** PyTorch, `accelerate`, `diffsynth` (this repo), `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-05-18-wan-vace-inpainting-design.md`

**Branch:** Work on `inpaint-design` (already checked out).

---

## File structure

**New files:**
- `diffsynth/utils/inpaint_mask.py` — mask-grid helpers (~80 lines)
- `tests/__init__.py` — empty marker
- `tests/test_inpaint_mask_utils.py` — unit tests for helpers
- `tests/test_vace_mask_slicing.py` — VACE forward with `mask_bool`
- `tests/test_inpaint_loss.py` — masked MSE loss
- `tests/test_inpaint_pipeline_unit.py` — `WanVideoUnit_InpaintMask` smoke test
- `tests/test_model_fn_inpaint.py` — `model_fn_wan_video` with inpaint flags
- `examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh`
- `examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B_inpaint_stage2.sh`
- `examples/wanvideo/model_training/lora/Wan2.1-VACE-14B_inpaint_stage1.sh`
- `examples/wanvideo/model_training/full/Wan2.1-VACE-14B_inpaint_stage2.sh`
- `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py`

**Modified files:**
- `diffsynth/models/wan_video_vace.py` — `VaceWanModel.forward` accepts `mask_bool`; `VaceWanAttentionBlock.forward` accepts optional `x_subset`
- `diffsynth/pipelines/wan_video.py` — `from_pretrained` accepts `enable_inpaint_global`, new `WanVideoUnit_InpaintMask`, `model_fn_wan_video` extended, `WanVideoPipeline.__call__` accepts `inpaint_latent_surgery`
- `diffsynth/diffusion/loss.py` — new `WanVideoInpaintMaskedLoss`
- `diffsynth/core/data/operators.py` — small additions if needed (probably none; mask loads via existing `LoadVideo` on a separate `data_file_keys` entry)
- `examples/wanvideo/model_training/train.py` — CLI flags, `WanTrainingModule` kwargs, freeze logic, `task_to_loss` entry

---

## Mask resolution conventions

Two `mask_bool` flavors exist; the pipeline unit emits both:

| Name | Shape | Spatial factor vs input | Used for |
|---|---|---|---|
| `mask_bool_latent` | `(T_lat * H/8 * W/8,)` bool | 8 (VAE latent grid) | Masked-MSE loss; latent surgery |
| `mask_bool_token` | `(T_lat * H/16 * W/16,)` bool | 16 (DiT patch grid) | VACE token slicing |

Where `T_lat = 1 + (T-1)/4` (Wan VAE temporal compression). `compute_downsampled_masks_tensor(mask, factor=8)` and `factor=16` produce them.

---

## Task 1: Set up test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Verify pytest is installed**

```bash
python -c "import pytest; print(pytest.__version__)"
```

Expected: prints a version. If `ModuleNotFoundError`, run `pip install pytest`.

- [ ] **Step 2: Create empty test package marker**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
# tests/conftest.py
import torch
import pytest

@pytest.fixture
def cpu_device():
    return torch.device("cpu")

@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)
```

- [ ] **Step 4: Verify pytest discovers the suite**

```bash
pytest tests/ -v
```

Expected: `collected 0 items` and exit code 0 (no tests yet).

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: scaffold tests directory with pytest config"
```

---

## Task 2: Implement `diffsynth/utils/inpaint_mask.py`

**Files:**
- Create: `diffsynth/utils/inpaint_mask.py`
- Create: `tests/test_inpaint_mask_utils.py`

The fork's `DiffSynthInpaint/diffsynth/pipelines/wan_video_new.py:30-106` has the reference implementation. Ported into a dedicated utility module.

- [ ] **Step 1: Write failing test for `downsample_tensor`**

```python
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
```

- [ ] **Step 2: Run and verify it fails**

```bash
pytest tests/test_inpaint_mask_utils.py::test_downsample_tensor_max_pool_factor_2 -v
```

Expected: `ModuleNotFoundError: No module named 'diffsynth.utils.inpaint_mask'`.

- [ ] **Step 3: Write additional failing tests**

Append to `tests/test_inpaint_mask_utils.py`:

```python
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
```

- [ ] **Step 4: Implement `diffsynth/utils/inpaint_mask.py`**

```python
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
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
pytest tests/test_inpaint_mask_utils.py -v
```

Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add diffsynth/utils/inpaint_mask.py tests/test_inpaint_mask_utils.py
git commit -m "feat(inpaint): add mask grid helpers (downsample, dilate)"
```

---

## Task 3: Add `mask_bool` token slicing to `VaceWanModel.forward`

**Files:**
- Modify: `diffsynth/models/wan_video_vace.py`
- Create: `tests/test_vace_mask_slicing.py`

Two sub-changes:
1. `VaceWanAttentionBlock.forward` accepts an optional `x_subset` for the `before_proj(c) + x` addition at block 0.
2. `VaceWanModel.forward` accepts `mask_bool`. When non-None, slice `c` to mask tokens, run the blocks on the short sequence, scatter `after_proj` outputs back into full-length zero buffers.

- [ ] **Step 1: Write failing test for backward-compat (mask_bool=None)**

```python
# tests/test_vace_mask_slicing.py
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


def test_vace_forward_mask_bool_none_matches_baseline_shape():
    m = _tiny_model().eval()
    B, seq, dim = 1, 12, 16
    x = torch.zeros(B, seq, dim)
    # vace_context: list of one tensor (vace_in_dim=8, F, H, W). With patch_size (1,2,2)
    # and F=1, H=4, W=6 -> patch grid (1, 2, 3) = 6 tokens. Zero-pad to seq=12.
    vace_context = [torch.zeros(8, 1, 4, 6)]
    context = torch.zeros(B, 1, dim)   # dummy text tokens
    t_mod = torch.zeros(B, 6, dim)     # adaLN modulation params
    freqs = torch.zeros(seq, 1, dim // 2)  # placeholder RoPE freqs
    hints = m(x, vace_context, context, t_mod, freqs)
    # Returns a tuple of length len(vace_layers); each hint matches x seq length
    assert len(hints) == 1
    assert hints[0].shape == (B, seq, dim)
```

- [ ] **Step 2: Run, expect it to error or fail because the model is being constructed without args you haven't checked**

```bash
pytest tests/test_vace_mask_slicing.py::test_vace_forward_mask_bool_none_matches_baseline_shape -v
```

If the test FAILS for reasons unrelated to mask slicing (e.g., placeholder freqs shape mismatch), adjust the tensor shapes to whatever the existing `VaceWanModel`/`DiTBlock` expects. Read `diffsynth/models/wan_video_dit.py:211-246` for `DiTBlock.forward` to confirm.

- [ ] **Step 3: Add failing test for `mask_bool` slicing behaviour**

Append to `tests/test_vace_mask_slicing.py`:

```python
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
    x = torch.randn(B, seq, dim)
    vace_context = [torch.randn(8, 1, 4, 6)]
    context = torch.zeros(B, 1, dim)
    t_mod = torch.zeros(B, 6, dim)
    freqs = torch.zeros(seq, 1, dim // 2)

    mask_bool = torch.zeros(seq, dtype=torch.bool)
    mask_bool[2:6] = True  # 4 active token positions

    hints = m(x, vace_context, context, t_mod, freqs, mask_bool=mask_bool)
    assert hints[0].shape == (B, seq, dim)
    # Positions outside mask must be zero
    outside = hints[0][:, ~mask_bool, :]
    assert torch.all(outside == 0), "hint values outside mask must be zero"
    # Positions inside mask must be non-zero (with positive-init weights and random x)
    inside = hints[0][:, mask_bool, :]
    assert torch.any(inside != 0), "hint values inside mask must be non-zero"
```

- [ ] **Step 4: Run, expect both tests to fail (forward signature doesn't accept `mask_bool`)**

```bash
pytest tests/test_vace_mask_slicing.py -v
```

Expected: TypeError for `mask_bool` kwarg.

- [ ] **Step 5: Modify `diffsynth/models/wan_video_vace.py`**

Replace the file contents with:

```python
# diffsynth/models/wan_video_vace.py
import torch
from .wan_video_dit import DiTBlock
from ..core.gradient import gradient_checkpoint_forward


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, x_subset=None):
        """If `x_subset` is given, it's the slice of `x` at the masked positions
        and is used in place of `x` for the block-0 `before_proj` residual.
        Downstream blocks still operate on the short `c` sequence."""
        if self.block_id == 0:
            c = self.before_proj(c) + (x_subset if x_subset is not None else x)
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        return torch.stack(all_c)


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        mask_bool: torch.Tensor = None,
    ):
        # 1. Patch-embed VACE context and flatten
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]   # each: (1, gp_seq, dim)

        if mask_bool is None:
            # Upstream behavior: zero-pad to DiT seq length
            c = torch.cat([
                torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1)
                for u in c
            ])
            x_subset = None
            scatter_back = False
        else:
            # Inpaint-local branch: slice to mask tokens
            assert mask_bool.dtype == torch.bool and mask_bool.dim() == 1, "mask_bool must be 1-D bool"
            assert mask_bool.numel() == x.shape[1], "mask_bool must match DiT seq length"
            # Truncate each c to x.shape[1] then slice
            c = torch.cat([u[:, : x.shape[1], :] for u in c])
            c = c[:, mask_bool, :]
            x_subset = x[:, mask_bool, :]
            scatter_back = True

        for block in self.vace_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs, x_subset,
            )

        # Unbind hints. With mask_bool, hints are at masked-position length;
        # scatter into full-length zero buffers so downstream code is unchanged.
        hints = torch.unbind(c)[:-1]
        if not scatter_back:
            return hints
        scattered = []
        for h in hints:
            buf = h.new_zeros(h.shape[0], x.shape[1], h.shape[-1])
            buf[:, mask_bool, :] = h
            scattered.append(buf)
        return tuple(scattered)
```

- [ ] **Step 6: Run tests, expect all pass**

```bash
pytest tests/test_vace_mask_slicing.py -v
```

Expected: 2 PASSED.

- [ ] **Step 7: Sanity-check upstream regression (mask_bool=None path bit-identical)**

```bash
pytest tests/test_vace_mask_slicing.py::test_vace_forward_mask_bool_none_matches_baseline_shape -v
```

(Already in the suite; just confirming.)

- [ ] **Step 8: Commit**

```bash
git add diffsynth/models/wan_video_vace.py tests/test_vace_mask_slicing.py
git commit -m "feat(inpaint): VaceWanModel mask_bool token slicing with scatter-back"
```

---

## Task 4: Add `WanVideoInpaintMaskedLoss`

**Files:**
- Modify: `diffsynth/diffusion/loss.py`
- Create: `tests/test_inpaint_loss.py`

The loss restricts MSE to masked latent positions. We'll factor out a `masked_token_mse(noise_pred, target, mask_bool_latent)` helper to keep it testable in isolation, and have `WanVideoInpaintMaskedLoss` use it.

- [ ] **Step 1: Note: `FlowMatchSFTLoss` is a function, not a class**

`diffsynth/diffusion/loss.py` defines `FlowMatchSFTLoss` as a function (line 5):

```python
def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    ...
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    ...
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
```

We'll mirror that shape — `WanVideoInpaintMaskedLoss` is also a function that delegates the bulk of the work to the same per-step machinery and substitutes the final MSE for `masked_token_mse`.

- [ ] **Step 2: Write failing test for `masked_token_mse`**

```python
# tests/test_inpaint_loss.py
import torch
import pytest

from diffsynth.diffusion.loss import masked_token_mse


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
    # MSE over (1 channel * 16 channels worth ... actually pred is (1,16,T,H,W).
    # Mask expanded over channels picks 16 positions, one of which differs by 1.
    # MSE = (1.0**2) / 16 = 0.0625
    assert torch.isclose(loss, torch.tensor(0.0625))
```

- [ ] **Step 3: Run, verify it fails**

```bash
pytest tests/test_inpaint_loss.py -v
```

Expected: `ImportError: cannot import name 'masked_token_mse'`.

- [ ] **Step 4: Append `masked_token_mse` and `WanVideoInpaintMaskedLoss` to `diffsynth/diffusion/loss.py`**

Add at the end of the file:

```python
import torch.nn.functional as _F   # already imported as torch.nn.functional elsewhere; alias avoids name collision


def masked_token_mse(noise_pred: torch.Tensor, target: torch.Tensor, mask_bool_latent: torch.Tensor) -> torch.Tensor:
    """MSE restricted to masked latent positions.

    Args:
        noise_pred / target: (B, C, T_lat, H_lat, W_lat).
        mask_bool_latent:    (T_lat * H_lat * W_lat,) bool. True where the
                             user-painted mask covers that latent token.

    Returns:
        Scalar mean-squared-error over masked positions across all channels.
    """
    B, C, T_lat, H_lat, W_lat = noise_pred.shape
    assert mask_bool_latent.dtype == torch.bool, "mask_bool_latent must be bool"
    assert mask_bool_latent.numel() == T_lat * H_lat * W_lat, "mask shape mismatch"
    m = mask_bool_latent.view(1, 1, T_lat, H_lat, W_lat).expand(B, C, T_lat, H_lat, W_lat)
    return _F.mse_loss(noise_pred[m], target[m])


def WanVideoInpaintMaskedLoss(pipe: BasePipeline, **inputs):
    """FlowMatch SFT loss restricted to masked latent tokens.

    Mirrors `FlowMatchSFTLoss` but substitutes `masked_token_mse` for the final
    MSE when `mask_bool_latent` is present in `inputs`.
    """
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    mask_bool_latent = inputs.get("mask_bool_latent")
    if mask_bool_latent is not None:
        loss = masked_token_mse(
            noise_pred.float(), training_target.float(),
            mask_bool_latent.to(noise_pred.device),
        )
    else:
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
```

(The body is a copy of `FlowMatchSFTLoss` with two changes: read `mask_bool_latent` from `inputs`, and swap the MSE for `masked_token_mse` when the mask is present. Falling back to vanilla MSE means even if the unit somehow produces `mask_bool_latent=None`, the loss is well-defined.)

- [ ] **Step 5: Run the helper tests, verify they pass**

```bash
pytest tests/test_inpaint_loss.py -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Smoke-test the `WanVideoInpaintMaskedLoss` class importable**

```python
# Append to tests/test_inpaint_loss.py
def test_wan_video_inpaint_masked_loss_importable():
    from diffsynth.diffusion.loss import WanVideoInpaintMaskedLoss
    assert WanVideoInpaintMaskedLoss is not None
```

Run:

```bash
pytest tests/test_inpaint_loss.py::test_wan_video_inpaint_masked_loss_importable -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add diffsynth/diffusion/loss.py tests/test_inpaint_loss.py
git commit -m "feat(inpaint): masked-token MSE loss for Wan-VACE inpainting"
```

---

## Task 5: Add `WanVideoUnit_InpaintMask` pipeline unit

**Files:**
- Modify: `diffsynth/pipelines/wan_video.py`
- Create: `tests/test_inpaint_pipeline_unit.py`

The unit reads `vace_video_mask` (binary mask frames) and `latents` (already-encoded input video, produced by `WanVideoUnit_InputVideoEmbedder` which runs earlier). It outputs `mask_bool_latent`, `mask_bool_token`, and `downsampled_input_latents` into `inputs_shared`.

- [ ] **Step 1: Read existing `WanVideoUnit_VACE` (line 649) and `PipelineUnit` base class to confirm the pipeline-unit pattern**

```bash
sed -n '649,712p' diffsynth/pipelines/wan_video.py
grep -n "class PipelineUnit" diffsynth/pipelines/base.py 2>/dev/null || \
  grep -rn "class PipelineUnit" diffsynth/ --include='*.py' | head -3
```

- [ ] **Step 2: Write failing test**

```python
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
    assert "downsampled_input_latents" not in out


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
    assert "downsampled_input_latents" in out
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
    assert out == {} or all(v is None for v in out.values())
```

- [ ] **Step 3: Run, verify failure**

```bash
pytest tests/test_inpaint_pipeline_unit.py -v
```

Expected: `ImportError: cannot import name 'WanVideoUnit_InpaintMask'`.

- [ ] **Step 4: Add `WanVideoUnit_InpaintMask` to `diffsynth/pipelines/wan_video.py`**

Insert immediately after the `WanVideoUnit_VACE` class (around line 712):

```python
class WanVideoUnit_InpaintMask(PipelineUnit):
    """Derive token-grid and latent-grid masks for inpainting, plus the
    downsampled input latents used by the Stage 2 global path.

    Runs after WanVideoUnit_VACE (which preprocesses vace_video_mask) and
    after WanVideoUnit_InputVideoEmbedder (which encodes `input_video` to
    `latents`). No-op when both inpaint flags are off.
    """

    def __init__(self):
        super().__init__(
            input_params=("vace_video_mask", "latents", "inpaint_local_enabled", "inpaint_global_enabled"),
            output_params=("mask_bool_latent", "mask_bool_token", "downsampled_input_latents"),
            onload_model_names=(),
        )

    def process(
        self,
        pipe,
        vace_video_mask,
        latents,
        inpaint_local_enabled=False,
        inpaint_global_enabled=False,
    ):
        if not (inpaint_local_enabled or inpaint_global_enabled):
            return {"mask_bool_latent": None, "mask_bool_token": None, "downsampled_input_latents": None}
        if vace_video_mask is None:
            return {"mask_bool_latent": None, "mask_bool_token": None, "downsampled_input_latents": None}

        from ..utils.inpaint_mask import compute_downsampled_masks_tensor
        import torch.nn.functional as F

        # vace_video_mask comes in as (B, 1, T, H, W) after preprocessing. We
        # operate on a single sample at a time (B=1 for now; B>1 left as a
        # follow-on -- see "Out of scope" in the spec).
        m = vace_video_mask[0]   # (1, T, H, W) -- C=1

        mask_latent = compute_downsampled_masks_tensor(m, factor=8)   # (1, T_lat, H/8, W/8)
        mask_token  = compute_downsampled_masks_tensor(m, factor=16)  # (1, T_lat, H/16, W/16)

        out = {
            "mask_bool_latent": mask_latent.flatten().to(torch.bool),
            "mask_bool_token":  mask_token.flatten().to(torch.bool),
            "downsampled_input_latents": None,
        }

        if inpaint_global_enabled:
            # Half-resolution view of the latents for the global path.
            out["downsampled_input_latents"] = F.interpolate(
                latents, scale_factor=(1.0, 0.5, 0.5), mode="area"
            )

        return out
```

Also: register the unit in `WanVideoPipeline.__init__`'s `self.units = [...]` list (starts at **line 55**, `WanVideoUnit_VACE()` is at **line 68**). Insert `WanVideoUnit_InpaintMask(),` on the line immediately after `WanVideoUnit_VACE(),`. The unit needs `latents` (produced by `WanVideoUnit_InputVideoEmbedder()` at line 60) and `vace_video_mask` (consumed by `WanVideoUnit_VACE()`); since both run before the new unit's slot, the inputs are ready.

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/test_inpaint_pipeline_unit.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add diffsynth/pipelines/wan_video.py tests/test_inpaint_pipeline_unit.py
git commit -m "feat(inpaint): WanVideoUnit_InpaintMask derives latent/token masks + downsampled latents"
```

---

## Task 6: Attach `global_patch_embedding` post-load in `from_pretrained`

**Files:**
- Modify: `diffsynth/pipelines/wan_video.py`

Add an `enable_inpaint_global` kwarg. After `pipe.dit` is loaded, deepcopy `patch_embedding` into `global_patch_embedding` and set a flag.

- [ ] **Step 1: Locate `from_pretrained` signature**

```bash
grep -n "def from_pretrained" diffsynth/pipelines/wan_video.py
```

Should be ~line 112.

- [ ] **Step 2: Add the kwarg**

In the `def from_pretrained(...)` signature, add `enable_inpaint_global: bool = False,` as the last positional kwarg (before `vram_limit` is fine).

- [ ] **Step 3: Add the post-load attachment**

After the line `pipe.vap = model_pool.fetch_model("wan_video_vap")` (or after `pipe.animate_adapter = ...`, whichever is the last DiT-related load), insert:

```python
# --- Inpainting global path (Stage 2 / inference) ---------------------
if enable_inpaint_global and pipe.dit is not None:
    import copy
    pipe.dit.global_patch_embedding = copy.deepcopy(pipe.dit.patch_embedding)
    pipe.dit.enable_inpaint_global = True
    if getattr(pipe, "dit2", None) is not None:
        pipe.dit2.global_patch_embedding = copy.deepcopy(pipe.dit2.patch_embedding)
        pipe.dit2.enable_inpaint_global = True
# ---------------------------------------------------------------------
```

- [ ] **Step 4: Smoke-test by importing the pipeline (no real model load)**

```bash
python -c "from diffsynth.pipelines.wan_video import WanVideoPipeline; import inspect; sig = inspect.signature(WanVideoPipeline.from_pretrained); assert 'enable_inpaint_global' in sig.parameters; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add diffsynth/pipelines/wan_video.py
git commit -m "feat(inpaint): from_pretrained kwarg + post-load global_patch_embedding attach"
```

---

## Task 7: Extend `model_fn_wan_video` with inpaint flags

**Files:**
- Modify: `diffsynth/pipelines/wan_video.py`
- Create: `tests/test_model_fn_inpaint.py`

The function gains four new kwargs:
- `mask_bool_token=None` — passed through to `vace(..., mask_bool=...)`.
- `downsampled_input_latents=None` — patch-embedded via `dit.global_patch_embedding` and concatenated into `context`.
- `inpaint_local_enabled=False` / `inpaint_global_enabled=False` — explicit gates.

- [ ] **Step 1: Locate `model_fn_wan_video`**

```bash
grep -n "def model_fn_wan_video" diffsynth/pipelines/wan_video.py
```

Should be ~line 1276.

- [ ] **Step 2: Write failing test**

```python
# tests/test_model_fn_inpaint.py
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
```

- [ ] **Step 3: Run, expect failure**

```bash
pytest tests/test_model_fn_inpaint.py -v
```

Expected: `AssertionError: missing kwarg: mask_bool_token`.

- [ ] **Step 4: Extend `model_fn_wan_video`**

In the signature, add (after the existing kwargs, before `**kwargs` if any):

```python
mask_bool_token: torch.Tensor = None,
downsampled_input_latents: torch.Tensor = None,
inpaint_local_enabled: bool = False,
inpaint_global_enabled: bool = False,
```

Inside the body, after `context = dit.text_embedding(context)` (**line 1394**), insert:

```python
# --- Inpainting global path: concatenate gp tokens into cross-attn ctx ---
if inpaint_global_enabled and downsampled_input_latents is not None:
    gp = dit.global_patch_embedding(downsampled_input_latents)   # (B, dim, f, h, w)
    gp = gp.flatten(2).transpose(1, 2)                            # (B, f*h*w, dim)
    context = torch.cat((context, gp), dim=1)                     # (B, ctx + gp_seq, dim)
# ------------------------------------------------------------------------
```

Then locate the VACE call (**line 1525-1527**). It currently looks like:

```python
if vace_context is not None:
    hints = vace(
        x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
```

Modify to pass `mask_bool` when local is enabled:

```python
if vace_context is not None:
    hints = vace(
        x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        mask_bool=mask_bool_token if inpaint_local_enabled else None,
    )
```

- [ ] **Step 5: Run smoke test**

```bash
pytest tests/test_model_fn_inpaint.py -v
```

Expected: PASS.

- [ ] **Step 6: Confirm `__call__` threads the new inputs through**

Inside `WanVideoPipeline.__call__`, find where `inputs_shared` is built (around the `WanVideoUnit_VACE` call site). Ensure `mask_bool_token`, `downsampled_input_latents`, `inpaint_local_enabled`, `inpaint_global_enabled` are all keys in `inputs_shared` (they will be auto-populated if the new `WanVideoUnit_InpaintMask` is in `self.units`). Then where `model_fn` is called (`noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)`), the new kwargs flow through `**inputs_shared`. No code change needed here as long as the unit is registered.

Add a quick guard near the top of `__call__`: accept `inpaint_local_enabled` and `inpaint_global_enabled` as kwargs (defaulting to False), and put them into `inputs_shared`. Without that, the inference call won't know to enable the units.

Find the `__call__` signature (**line 190**) and add:

```python
inpaint_local_enabled: bool = False,
inpaint_global_enabled: bool = False,
inpaint_latent_surgery: bool = False,   # used in Task 8
```

And in the body where `inputs_shared = {...}` is constructed (**line 284**), add the three keys:

```python
        inputs_shared = {
            ...,
            "inpaint_local_enabled": inpaint_local_enabled,
            "inpaint_global_enabled": inpaint_global_enabled,
            "inpaint_latent_surgery": inpaint_latent_surgery,
        }
```

- [ ] **Step 7: Commit**

```bash
git add diffsynth/pipelines/wan_video.py tests/test_model_fn_inpaint.py
git commit -m "feat(inpaint): extend model_fn_wan_video with mask-token + global-context paths"
```

---

## Task 8: Add `inpaint_latent_surgery` to `WanVideoPipeline.__call__`

**Files:**
- Modify: `diffsynth/pipelines/wan_video.py`

After each scheduler step in the denoising loop, when `inpaint_latent_surgery=True` and `mask_bool_latent` is available, replace outside-mask latents with the noised original.

- [ ] **Step 1: The denoising loop step is at line 334**

`inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])` — that's the per-step update. We add the surgery immediately after that line.

`WanVideoUnit_InputVideoEmbedder` (line 400) produces `inputs_shared["input_latents"]` (the un-noised VAE encoding of the input video). We re-noise that to the next timestep for outside-mask pinning. A `noise` tensor is sampled inside the loss function but not stored in `inputs_shared` — we'll sample a fresh one here for the surgery (the slight noise pattern mismatch is fine since it only affects outside-mask, which we discard).

- [ ] **Step 2: Insert the surgery block after line 334**

```python
# --- Inpaint latent surgery: outside-mask latents pinned to noisy original ---
if inpaint_latent_surgery and inputs_shared.get("mask_bool_latent") is not None and inputs_shared.get("input_latents") is not None:
    mb = inputs_shared["mask_bool_latent"]
    latents = inputs_shared["latents"]
    T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
    mask = mb.view(1, 1, T_lat, H_lat, W_lat).expand_as(latents).to(latents.device)
    if progress_id + 1 < len(self.scheduler.timesteps):
        t_next = self.scheduler.timesteps[progress_id + 1]
        noise = torch.randn_like(inputs_shared["input_latents"])
        original_noised = self.scheduler.add_noise(inputs_shared["input_latents"], noise, t_next)
        inputs_shared["latents"] = torch.where(mask, latents, original_noised)
# ----------------------------------------------------------------------------
```

(Note: the surgery mutates `inputs_shared["latents"]` rather than a local `latents` variable, since the next iteration reads `inputs_shared["latents"]`.)

- [ ] **Step 3: Smoke-test that the kwarg is accepted**

```bash
python -c "
import inspect
from diffsynth.pipelines.wan_video import WanVideoPipeline
sig = inspect.signature(WanVideoPipeline.__call__)
assert 'inpaint_latent_surgery' in sig.parameters
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add diffsynth/pipelines/wan_video.py
git commit -m "feat(inpaint): per-step outside-mask latent surgery in WanVideoPipeline.__call__"
```

---

## Task 9: Dataset support for `mask` column

**Files:**
- Modify: `diffsynth/core/data/operators.py` (likely a small extension)
- Modify: `examples/wanvideo/model_training/train.py` (already in Task 10, but reference here)

The fork's `VideoInpaintingDataset` loaded paired video + mask. Upstream's `UnifiedDataset` already supports multiple file keys via `--data_file_keys`. So loading a `mask` column should "just work" by treating it like any other video file. Verify and add a single-channel preprocessing knob if needed.

- [ ] **Step 1: Confirm `LoadVideo` operator works on a single-channel mask video**

```bash
grep -n "class LoadVideo\|def __call__\|num_channels\|RGB" diffsynth/core/data/operators.py | head -20
```

If `LoadVideo` always converts to RGB, masks will be 3-channel duplicates — fine for our purposes (we max across channels in `compute_downsampled_masks_tensor` anyway, but only the first channel is used since `C` in that function is treated per-channel).

- [ ] **Step 2: If needed, add an explicit single-channel path**

If `LoadVideo` returns 3-channel and we want 1-channel masks, append to `operators.py`:

```python
class VideoToSingleChannel:
    """Convert a (T, 3, H, W) RGB mask video to (T, 1, H, W) by taking channel 0."""
    def __call__(self, sample):
        sample = sample.clone()
        for key in list(sample.keys()):
            if key.startswith("mask") and sample[key].ndim == 4 and sample[key].shape[1] == 3:
                sample[key] = sample[key][:, :1]
        return sample
```

Otherwise skip this step.

- [ ] **Step 3: Verify operator import works**

```bash
python -c "from diffsynth.core.data.operators import LoadVideo; print(LoadVideo)"
```

Expected: prints the class.

- [ ] **Step 4: Commit**

```bash
git add diffsynth/core/data/operators.py
git commit -m "feat(inpaint): single-channel mask passthrough in dataset operators"
```

(If no change was needed, skip this commit.)

---

## Task 10: Extend `train.py` with inpainting flags + freeze logic

**Files:**
- Modify: `examples/wanvideo/model_training/train.py`

Add three flags, a new task value, kwargs to `WanTrainingModule`, the Stage-2 freeze pass, and the inpainting input synthesis.

- [ ] **Step 1: Add flags to `wan_parser`**

In `def wan_parser()` (around line 114), before `return parser`:

```python
    parser.add_argument("--enable_inpaint_local", action="store_true",
                        help="Enable token-level mask slicing in VACE (Stage 1+).")
    parser.add_argument("--enable_inpaint_global", action="store_true",
                        help="Enable downsampled-input-latent tokens in cross-attn context (Stage 2).")
    parser.add_argument("--stage2_freeze", action="store_true",
                        help="Freeze all cross-attn then unfreeze k_img/v_img/zero_proj/global_patch_embedding (Stage 2).")
```

- [ ] **Step 2: Plumb the kwargs into `WanTrainingModule.__init__`**

Find the `WanTrainingModule(__init__)` signature (line 10). Add to the kwargs:

```python
        enable_inpaint_local: bool = False,
        enable_inpaint_global: bool = False,
        stage2_freeze: bool = False,
```

Inside `__init__`, modify the pipe construction:

```python
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            enable_inpaint_global=enable_inpaint_global,   # NEW
        )
```

After `self.switch_pipe_to_training_mode(...)`, add the freeze pass:

```python
        if stage2_freeze:
            for name, p in self.pipe.named_parameters():
                if "cross_attn" in name:
                    p.requires_grad = False
            for name, p in self.pipe.named_parameters():
                if any(k in name for k in ("k_img_norm", "v_img", "zero_proj", "k_img", "global_patch_embedding")):
                    p.requires_grad = True
            trainable = [n for n, p in self.pipe.named_parameters() if p.requires_grad]
            print(f"[stage2_freeze] {len(trainable)} trainable params; sample: {trainable[:5]}")

        self.enable_inpaint_local = enable_inpaint_local
        self.enable_inpaint_global = enable_inpaint_global
```

- [ ] **Step 3: Add the `sft:inpaint` task entry**

In the `self.task_to_loss` dict, add:

```python
            "sft:inpaint": lambda pipe, inputs_shared, inputs_posi, inputs_nega:
                WanVideoInpaintMaskedLoss(pipe, **inputs_shared, **inputs_posi),
```

You'll also need to import `WanVideoInpaintMaskedLoss` at the top of `train.py`:

```python
from diffsynth.diffusion.loss import WanVideoInpaintMaskedLoss
```

(If the existing `from diffsynth.diffusion import *` already covers it, no extra import needed — confirm.)

- [ ] **Step 4: Inject the inpaint flags into `get_pipeline_inputs`**

Modify `get_pipeline_inputs` to set `inpaint_local_enabled` and `inpaint_global_enabled` in `inputs_shared`. After the existing `inputs_shared = {...}` block:

```python
        if self.enable_inpaint_local or self.enable_inpaint_global:
            inputs_shared["inpaint_local_enabled"] = self.enable_inpaint_local
            inputs_shared["inpaint_global_enabled"] = self.enable_inpaint_global
            # Synthesize vace_video from `video` + `mask` if not already set
            if "vace_video" not in data and "mask" in data:
                data["vace_video"] = [v.masked_fill(m > 0.5, 0)
                                       for v, m in zip(data["video"], data["mask"])]
                data["vace_video_mask"] = data["mask"]
```

- [ ] **Step 5: Plumb the new kwargs into the `WanTrainingModule(...)` construction call**

In the `if __name__ == "__main__":` block, where `model = WanTrainingModule(...)` is built, add the three new args:

```python
        enable_inpaint_local=args.enable_inpaint_local,
        enable_inpaint_global=args.enable_inpaint_global,
        stage2_freeze=args.stage2_freeze,
```

- [ ] **Step 6: Plumb `sft:inpaint` into the `launcher_map`**

Add an entry so `--task sft:inpaint` is routed correctly:

```python
    launcher_map = {
        ...,
        "sft:inpaint": launch_training_task,
        "sft:inpaint:data_process": launch_data_process_task,   # if/when data-preprocess is needed
    }
```

(Add at least `"sft:inpaint": launch_training_task`. Skip the `:data_process` entry if not needed.)

- [ ] **Step 7: Smoke-test the CLI**

```bash
python examples/wanvideo/model_training/train.py --help 2>&1 | grep -E "enable_inpaint|stage2_freeze"
```

Expected: three lines listing the new flags.

- [ ] **Step 8: Commit**

```bash
git add examples/wanvideo/model_training/train.py
git commit -m "feat(inpaint): train.py flags + WanTrainingModule freeze + sft:inpaint task"
```

---

## Task 11: Stage 1 training shell scripts

**Files:**
- Create: `examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh`
- Create: `examples/wanvideo/model_training/lora/Wan2.1-VACE-14B_inpaint_stage1.sh`

- [ ] **Step 1: Create 1.3B Stage 1 launcher**

```bash
# examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh
export HF_HOME="/grogu/user/ylitman/.cache"
accelerate launch examples/wanvideo/model_training/train.py \
  --task "sft:inpaint" \
  --dataset_base_path /grogu/user/ylitman/datasets/VPData/videovo/raw_video \
  --dataset_metadata_path /grogu/user/ylitman/datasets/VPData/pexels_videovo_train_dataset.csv \
  --data_file_keys "video,mask" \
  --height 480 --width 720 --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --max_train_steps 8000 \
  --save_steps 500 \
  --gradient_accumulation_steps 8 \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_inpaint_stage1" \
  --enable_inpaint_local \
  --use_gradient_checkpointing_offload
```

- [ ] **Step 2: Create 14B Stage 1 launcher**

Copy the 1.3B version, change three things: `Wan2.1-VACE-1.3B` → `Wan2.1-VACE-14B`, `gradient_accumulation_steps 8` → `1`, output path suffix → `Wan2.1-VACE-14B_inpaint_stage1`. Add `--use_gradient_checkpointing` (the 14B model needs it).

- [ ] **Step 3: Make both executable**

```bash
chmod +x examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh
chmod +x examples/wanvideo/model_training/lora/Wan2.1-VACE-14B_inpaint_stage1.sh
```

- [ ] **Step 4: Commit**

```bash
git add examples/wanvideo/model_training/lora/Wan2.1-VACE-*_inpaint_stage1.sh
git commit -m "feat(inpaint): Stage 1 LoRA training scripts (1.3B + 14B)"
```

---

## Task 12: Stage 2 training shell scripts

**Files:**
- Create: `examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B_inpaint_stage2.sh`
- Create: `examples/wanvideo/model_training/full/Wan2.1-VACE-14B_inpaint_stage2.sh`

- [ ] **Step 1: Create 1.3B Stage 2 launcher**

```bash
# examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B_inpaint_stage2.sh
export HF_HOME="/grogu/user/ylitman/.cache"
accelerate launch examples/wanvideo/model_training/train.py \
  --task "sft:inpaint" \
  --dataset_base_path /grogu/user/ylitman/datasets/VPData/videovo/raw_video \
  --dataset_metadata_path /grogu/user/ylitman/datasets/VPData/pexels_videovo_train_dataset.csv \
  --data_file_keys "video,mask" \
  --height 480 --width 720 --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --max_train_steps 4000 \
  --save_steps 250 \
  --gradient_accumulation_steps 1 \
  --train_batch_size 1 \
  --trainable_models "dit" \
  --preset_lora_path "./models/train/Wan2.1-VACE-1.3B_inpaint_stage1/step-8000.safetensors" \
  --preset_lora_model "vace" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_inpaint_stage2" \
  --enable_inpaint_local \
  --enable_inpaint_global \
  --stage2_freeze \
  --use_gradient_checkpointing_offload
```

- [ ] **Step 2: Create 14B Stage 2 launcher**

Same as 1.3B but with `Wan2.1-VACE-14B` everywhere. Path for `--preset_lora_path` should point at the 14B Stage 1 output.

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x examples/wanvideo/model_training/full/Wan2.1-VACE-*_inpaint_stage2.sh
git add examples/wanvideo/model_training/full/Wan2.1-VACE-*_inpaint_stage2.sh
git commit -m "feat(inpaint): Stage 2 DiT training scripts (1.3B + 14B)"
```

---

## Task 13: Inference script

**Files:**
- Create: `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py`

- [ ] **Step 1: Create the script**

```python
# examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py
"""Wan 2.1 VACE inpainting inference: Stage-1 LoRA + Stage-2 DiT delta.

Usage:
    python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py \
        --input_video path/to/input.mp4 \
        --input_mask  path/to/mask.mp4 \
        --prompt "a cat sitting on a chair" \
        --stage1_ckpt models/train/Wan2.1-VACE-1.3B_inpaint_stage1/step-8000.safetensors \
        --stage2_ckpt models/train/Wan2.1-VACE-1.3B_inpaint_stage2/step-4000.safetensors \
        --output_path inpainted.mp4
"""
import argparse
import torch
from diffsynth import load_state_dict, save_video, VideoData
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", required=True)
    p.add_argument("--input_mask",  required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", required=True)
    p.add_argument("--output_path", default="inpainted.mp4")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width",  type=int, default=720)
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        enable_inpaint_global=True,
    )

    # Stage 1: VACE LoRA
    pipe.load_lora(pipe.vace, args.stage1_ckpt)
    # Stage 2: DiT delta (global_patch_embedding + cross-attn proj subsets)
    pipe.dit.load_state_dict(load_state_dict(args.stage2_ckpt), strict=False)

    video = VideoData(args.input_video, height=args.height, width=args.width).read_video()[: args.num_frames]
    mask  = VideoData(args.input_mask,  height=args.height, width=args.width).read_video()[: args.num_frames]

    # vace_video = video with mask region zeroed
    vace_video = [
        torch.from_numpy(__import__("numpy").array(v)).clone() for v in video
    ]
    # The pipeline's preprocess_video accepts PIL list; pass through as-is.
    vace_video = video                              # zeroing happens inside WanVideoUnit_VACE
    vace_video_mask = mask

    out = pipe(
        prompt=args.prompt,
        input_video=video, denoising_strength=1.0,
        vace_video=vace_video, vace_video_mask=vace_video_mask,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        inpaint_local_enabled=True,
        inpaint_global_enabled=True,
        inpaint_latent_surgery=True,
    )

    save_video(out, args.output_path, fps=args.fps)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
```

> **Note for implementer:** double-check that `WanVideoUnit_VACE` performs the `inactive = vace_video * (1 - vace_video_mask)` zeroing internally (it does — see `diffsynth/pipelines/wan_video.py:676`). If so, you don't need to pre-zero `vace_video` in the script; pass the raw video and the mask, and the unit will do the rest.

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "import examples.wanvideo.model_inference.Wan2.1-VACE-1.3B_inpainting" 2>&1 | tail -5
```

(If the dotted filename causes an import error, just run `python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py --help` instead.)

```bash
python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py --help
```

Expected: prints argparse help.

- [ ] **Step 3: Commit**

```bash
git add examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py
git commit -m "feat(inpaint): Wan 2.1 VACE 1.3B inpainting inference script"
```

---

## Task 14: Manual GPU validation

**Files:**
- Create: `docs/superpowers/plans/wan-vace-inpaint-manual-validation.md`

These steps are run by the user on their GPU machine. The plan documents them so the implementer (and future-you) has a clear checklist.

- [ ] **Step 1: Create the validation checklist file**

```markdown
# Wan 2.1 VACE Inpainting — Manual Validation Checklist

Run after the automated tests in Tasks 1-10 pass.

## Stage 1 dry-run (1.3B)

- [ ] Run `bash examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh` for ≥100 steps.
- [ ] Confirm: loss prints decrease over time (no NaNs).
- [ ] Confirm: a checkpoint exists at `models/train/Wan2.1-VACE-1.3B_inpaint_stage1/step-500.safetensors`.
- [ ] Inspect with `python -c "from safetensors.torch import load_file; sd = load_file('...'); print(list(sd.keys())[:5])"` — keys should be VACE LoRA module names (e.g. `vace_blocks.0.self_attn.q.lora_A.default.weight`).

## Stage 2 dry-run (1.3B)

- [ ] Edit `Wan2.1-VACE-1.3B_inpaint_stage2.sh` so `--preset_lora_path` points at the Stage 1 checkpoint.
- [ ] Run for ≥100 steps.
- [ ] Confirm: loss prints (smaller than Stage 1 since fewer parameters), no NaNs.
- [ ] Checkpoint exists at `models/train/Wan2.1-VACE-1.3B_inpaint_stage2/step-250.safetensors`.
- [ ] Key prefixes in checkpoint should be `global_patch_embedding.*` and `blocks.*.cross_attn.{k_img,k_img_norm,v_img,zero_proj}.*`, and ONLY those.

## Inference smoke

- [ ] Pick one VPData clip (e.g., `eval_set/clip_001/video.mp4` + `mask.mp4`).
- [ ] Run:
  ```bash
  python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py \
      --input_video <clip.mp4> --input_mask <mask.mp4> --prompt "..." \
      --stage1_ckpt .../inpaint_stage1/step-8000.safetensors \
      --stage2_ckpt .../inpaint_stage2/step-4000.safetensors \
      --output_path /tmp/inpainted.mp4
  ```
- [ ] Confirm output video is saved and is visually different from the masked-out input *inside* the mask region.
- [ ] Confirm output video is identical (or near-identical) to the original input *outside* the mask region (latent surgery working).

## Upstream regression

- [ ] Run any existing Wan-VACE training/inference script (e.g. `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py`) WITHOUT inpaint flags.
- [ ] Confirm output is bit-identical (or numerically close — tolerate fp16 jitter) to a baseline produced before the inpainting changes.

## Sign-off

- [ ] All boxes above checked. Inpainting port is ready for paper/production use.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/wan-vace-inpaint-manual-validation.md
git commit -m "docs(inpaint): manual GPU validation checklist"
```

---

## Self-review (engineer)

After finishing all 14 tasks:

1. **Run the full automated suite:**
   ```bash
   pytest tests/ -v
   ```
   Expected: all tests in `test_inpaint_*` and `test_vace_*` pass.

2. **Confirm upstream is unchanged when flags are off:**
   ```bash
   python -c "
   import torch
   from diffsynth.models.wan_video_vace import VaceWanModel
   m = VaceWanModel(vace_layers=(0,), vace_in_dim=8, patch_size=(1,2,2),
                    dim=16, num_heads=2, ffn_dim=32)
   x = torch.zeros(1, 12, 16)
   vc = [torch.zeros(8, 1, 4, 6)]
   ctx = torch.zeros(1, 1, 16)
   t_mod = torch.zeros(1, 6, 16)
   freqs = torch.zeros(12, 1, 8)
   # Default call (no mask_bool) must produce same shape as upstream
   h = m(x, vc, ctx, t_mod, freqs)
   assert h[0].shape == (1, 12, 16)
   print('Upstream-shape OK')
   "
   ```

3. **Sanity-check the `inpaint-design` branch log:**
   ```bash
   git log --oneline inpaint-design ^main | wc -l
   ```
   Expected: 14-16 commits (one per task plus initial spec).

4. **Run the manual validation checklist** (Task 14 file).

---

## Out of scope (documented; do NOT implement in this round)

- Wan 2.2 VACE (`vace2`, `dit2`).
- Batch size B > 1 for inpainting (the pipeline unit operates on `vace_video_mask[0]` only).
- DAVIS / VPBench evaluation harness.
- Bbox controller (dead code in the fork).
- Decoder LoRA stage.
- Mask dilation as a CLI flag (helper is in `inpaint_mask.py` but not wired).
