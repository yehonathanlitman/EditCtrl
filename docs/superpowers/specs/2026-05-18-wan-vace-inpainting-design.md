# Wan 2.1 VACE Inpainting — Local + Global Token Conditioning

**Status:** Design
**Date:** 2026-05-18
**Scope:** Port the two-stage token-level latent-mask inpainting mechanism from `DiffSynthInpaint` into fresh `diffsynth-studio/`, for Wan 2.1 VACE (1.3B + 14B).

## Goal

Re-implement the inpainting conditioning used in `DiffSynthInpaint`'s `wan_video_new.py` so it lives in fresh upstream `diffsynth-studio` as small flag-gated additions. Produce:

1. Stage-1 training script — trains a VACE LoRA using only the **local** token-mask path (mask-token slicing in VACE; loss restricted to masked latent tokens).
2. Stage-2 training script — loads frozen Stage-1 LoRA, then trains the DiT's `global_patch_embedding` + cross-attn `k_img`/`v_img`/`zero_proj` projections, adding a **global** path that concatenates downsampled-input-latent tokens into the cross-attn context.
3. Inference script — applies both stages plus per-step outside-mask latent surgery.

Default upstream behavior must be unchanged with all inpainting flags off.

## Background

The fork's mechanism (see `DiffSynthInpaint/diffsynth/{pipelines/wan_video_new.py, models/wan_video_vace.py, models/wan_video_dit.py}`):

- **Local path** — VACE control sequence `c = vace_patch_embedding(vace_context)` is sliced to mask tokens: `c = c[:, mask_bool, :]`. Only mask tokens flow through `vace_blocks` and contribute residual hints back into the DiT.
- **Global path** — A second `nn.Conv3d` `global_patch_embedding` in `WanModel`, initialized as a deep-copy of `patch_embedding`. It patch-embeds a downsampled view of the full input latents; the resulting tokens are concatenated into the cross-attention context so the DiT attends to a global scene view while predicting in-mask latents.
- **Training curriculum** — Stage 1 trains VACE LoRA with the global path commented out (`context_w_global = context`). Stage 2 loads frozen Stage-1 LoRA, unfreezes `global_patch_embedding` + cross-attn projections.
- **Loss** — `F.mse_loss(noise_pred[mask_bool_expanded], target[mask_bool_expanded])`.
- **Inference latent surgery** — outside-mask latents are pinned to the noised original each step; only inside-mask predictions are kept.

## Scope decisions

| Decision | Choice | Notes |
|---|---|---|
| Target models | Wan 2.1 VACE (1.3B + 14B) only | Wan 2.2 VACE deferred. |
| Code location | In-place edits to `diffsynth-studio/` behind feature flags | Mirrors upstream WanToDance pattern. |
| Mask input | Reuse upstream `vace_video` + `vace_video_mask` | Existing `WanVideoUnit_VACE` already handles this format. |
| Dataset | VPData-style CSV with `video` + `mask` columns | Matches fork data layout. |
| Inference | Minimal inference script with latent surgery | DAVIS/VPBench evaluation deferred. |
| Approach | A — Feature flags on existing classes | Smallest diff vs upstream; matches WanToDance precedent. |

## Naming

- `global_patch_embedding` — new attribute on `WanModel`. Distinct from upstream's `patch_embedding_global` (used by WanToDance under `prepare_wantodance(wantodance_enable_unimodel=True)`). No collision.

## Per-file changes

### `diffsynth/utils/inpaint_mask.py` (NEW, ~80 lines)

Helpers ported from `DiffSynthInpaint/diffsynth/pipelines/wan_video_new.py:40-106`:

- `downsample_tensor(tensor, factor)` — 2D max-pool.
- `compute_downsampled_masks_tensor(masks, factor=8)` — input `(C, N, H, W)`; output `(C, M, H/factor, W/factor)` where `M = 1 + (N-1)/4`. First frame downsampled directly; subsequent frames in chunks of 4 reduced by max-union then spatially max-pooled — matches Wan VAE temporal compression.
- `dilate_mask(mask, dilation_radius)` — square-kernel binary dilation.

### `diffsynth/models/wan_video_vace.py` (74 → ~100 lines)

`VaceWanModel.forward` accepts a new optional `mask_bool: torch.Tensor = None` arg of shape `(seq_len,)` boolean. Branch:

- `mask_bool is None` → existing zero-pad-to-DiT-seq behavior (bit-identical to upstream).
- `mask_bool is not None` → after `c = vace_patch_embedding(...).flatten(2).transpose(1, 2)`, do `c = c[:, mask_bool, :]`. The block math is adjusted so:
  - `VaceWanAttentionBlock.forward` accepts an optional `x_subset` (the masked-position slice of `x`); at block 0, `c = before_proj(c) + x_subset`.
  - `after_proj` outputs are scattered back into a zero-filled buffer of shape `(B, full_seq, dim)` at `mask_bool` indices, yielding `hints` consumable by the DiT blocks unchanged.

### `diffsynth/models/wan_video_dit.py` — unchanged

`WanModel` itself is not modified. `WanModel` is constructed via the model registry inside `download_and_load_models`, so adding a kwarg to `__init__` would require touching the registry too. Instead, `global_patch_embedding` is attached after load (see pipeline section below). The DiT class stays clean; the inpainting attachment is purely a pipeline-level concern.

`WanModel.forward` is unchanged. The global path is applied from `model_fn_wan_video`.

### `diffsynth/pipelines/wan_video.py` (1717 → ~1820 lines)

**(a) `from_pretrained`** gains an `enable_inpaint_global: bool = False` kwarg. After `pipe.dit` is loaded, if `enable_inpaint_global` is True:

```python
import copy
pipe.dit.global_patch_embedding = copy.deepcopy(pipe.dit.patch_embedding)
pipe.dit.enable_inpaint_global = True
```

The deep-copy preserves the trained patch embedding weights as the starting point for the global path's Conv3d (matches the fork's `wan_video_new.py:542`). If Wan 2.2 is ever added, this also runs on `pipe.dit2`.

**(b) New `WanVideoUnit_InpaintMask` pipeline unit**, inserted after `WanVideoUnit_VACE` and `WanVideoUnit_VideoEmbedding`. Inputs: `vace_video_mask`, `input_latents`, `inpaint_local_enabled`, `inpaint_global_enabled`. Outputs into `inputs_shared`:

- `mask_bool` — `compute_downsampled_masks_tensor(vace_video_mask, factor=8).flatten().bool()` (only when `inpaint_local_enabled`).
- `downsampled_input_latents` — `F.interpolate(input_latents, scale_factor=0.5, mode='area')` (only when `inpaint_global_enabled`).

When both flags are False the unit is a no-op.

**(c) `model_fn_wan_video` extensions.** Accept new kwargs `mask_bool=None`, `downsampled_input_latents=None`, `inpaint_local_enabled=False`, `inpaint_global_enabled=False`. Two added branches inside:

```python
# After dit.patch_embedding and dit.text_embedding:
if inpaint_global_enabled:
    gp = dit.global_patch_embedding(downsampled_input_latents)   # (B, dim, f, h, w)
    gp = gp.flatten(2).transpose(1, 2)                            # (B, f*h*w, dim)
    context = torch.cat((context, gp), dim=1)                     # (B, ctx + gp_seq, dim)

# At the VACE call:
if vace_context is not None:
    hints = vace(x, vace_context, context, t_mod, freqs,
                 mask_bool=mask_bool if inpaint_local_enabled else None,
                 use_gradient_checkpointing=..., use_gradient_checkpointing_offload=...)
```

Note: the fork's `gp` construction iterated per batch and reshaped via `context.shape[-1]` — that path only handles `B=1` correctly. The version above is batch-aware (`B*f*h*w` tokens, then batched concat). This is a fix-up vs the fork.

`WanVideoPipeline.__call__` gains an `inpaint_latent_surgery: bool = False` kwarg. When True and a `vace_video_mask` is provided, after each scheduler step:

```python
T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
latent_mask = mask_bool.view(1, 1, T_lat, H_lat, W_lat).expand_as(latents)
latents = torch.where(latent_mask, latents, scheduler.add_noise(input_latents, noise, t_next))
```

Latent surgery interacts cleanly with `denoising_strength < 1.0`: the outside-mask region is pinned to the appropriately-noised original at the *current* timestep, so a partial denoise still preserves outside-mask content as expected.

The kwarg propagates as part of `inputs_shared` to the denoising loop. Default False (off for non-inpainting use).

### `diffsynth/diffusion/` — new `WanVideoInpaintMaskedLoss`

Mirrors `FlowMatchSFTLoss` but restricts the MSE to masked latent tokens:

```python
class WanVideoInpaintMaskedLoss(FlowMatchSFTLoss):
    # noise_pred / target are (B, 16, T_lat, H_lat, W_lat).
    # mask_bool is (T_lat * H_lat * W_lat,) bool.
    def __call__(self, pipe, *, mask_bool, **kwargs):
        # ... existing FlowMatchSFTLoss machinery to produce noise_pred + target ...
        T_lat, H_lat, W_lat = noise_pred.shape[2], noise_pred.shape[3], noise_pred.shape[4]
        m = mask_bool.view(1, 1, T_lat, H_lat, W_lat).expand_as(noise_pred)
        return F.mse_loss(noise_pred[m], target[m]) * weight
```

Registered into `WanTrainingModule.task_to_loss` under `"sft:inpaint"`.

### `diffsynth/core/data/operators.py` — `LoadVideoMaskPair` operator

Loads paired video + mask tracks. The mask track is a separate file (single-channel video, same frame count and frame rate as the video). Emits both into the sample dict under `video` and `mask`. Used via `--data_file_keys "video,mask"`.

### `examples/wanvideo/model_training/train.py` — extended

`wan_parser()` gains:

- `--enable_inpaint_local` (store_true)
- `--enable_inpaint_global` (store_true)
- `--stage2_freeze` (store_true)

`WanTrainingModule.__init__` gains kwargs `enable_inpaint_local=False, enable_inpaint_global=False, stage2_freeze=False`:

- Passes `enable_inpaint_global` into `WanVideoPipeline.from_pretrained`.
- If `stage2_freeze`:
  ```python
  for name, p in self.pipe.named_parameters():
      if "cross_attn" in name:
          p.requires_grad = False
  for name, p in self.pipe.named_parameters():
      if any(k in name for k in ("k_img_norm", "v_img", "zero_proj", "k_img", "global_patch_embedding")):
          p.requires_grad = True
  ```

`WanTrainingModule.get_pipeline_inputs`: when `enable_inpaint_local` or `enable_inpaint_global` are True, synthesize `data["vace_video"] = [v.masked_fill(m>0.5, 0) for v, m in zip(data["video"], data["mask"])]` and `data["vace_video_mask"] = data["mask"]` if not already present; thread `inpaint_local_enabled` / `inpaint_global_enabled` into `inputs_shared`.

`task_to_loss["sft:inpaint"]` → `WanVideoInpaintMaskedLoss`.

All additions are gated; with all flags off and `task != "sft:inpaint"`, behavior is identical to upstream.

### Shell scripts

**`examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B_inpaint_stage1.sh`** — see Section 4 of the brainstorm. Key flags: `--task "sft:inpaint" --enable_inpaint_local --lora_base_model "vace" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 128 --remove_prefix_in_ckpt "pipe.vace."`.

**`examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B_inpaint_stage2.sh`** — Key flags: `--task "sft:inpaint" --enable_inpaint_local --enable_inpaint_global --stage2_freeze --trainable_models "dit" --preset_lora_path <stage1.safetensors> --preset_lora_model "vace" --remove_prefix_in_ckpt "pipe.dit."`.

A `lora/Wan2.1-VACE-14B_inpaint_stage1.sh` / `full/Wan2.1-VACE-14B_inpaint_stage2.sh` pair are mechanical copies with the 14B `model_id_with_origin_paths` and adjusted `gradient_accumulation_steps`.

### Inference script

**`examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_inpainting.py`** — ~200 lines. Loads pipeline with `enable_inpaint_global=True`, applies Stage-1 VACE LoRA via `pipe.load_lora(pipe.vace, ...)`, loads Stage-2 DiT delta via `pipe.dit.load_state_dict(..., strict=False)`. Calls `pipe(..., inpaint_local_enabled=True, inpaint_global_enabled=True, inpaint_latent_surgery=True, ...)`.

## Stage activation matrix

| Param | Stage 1 | Stage 2 | Inference |
|---|---|---|---|
| `enable_inpaint_local` | True | True | True |
| `enable_inpaint_global` | False | True | True |
| `stage2_freeze` | — | True | — |
| `inpaint_latent_surgery` | — | — | True |
| Trainable | VACE LoRA (q,k,v,o,ffn.0,ffn.2) | DiT `global_patch_embedding` + `k_img`/`v_img`/`zero_proj`/`k_img_norm` | — |
| Loss | `sft:inpaint` (masked MSE) | `sft:inpaint` (masked MSE) | — |
| Continues from | Base Wan-VACE | Stage-1 LoRA via `--preset_lora_path` | Stage 1 LoRA + Stage 2 DiT delta |

## Data flow per stage

(See brainstorm Section 3 for the full step-by-step.)

**Stage 1.** Dataset emits `video` + `mask`. Trainer synthesizes `vace_video` and feeds it through upstream's `WanVideoUnit_VACE`. `WanVideoUnit_InpaintMask` produces `mask_bool`. `model_fn_wan_video` runs with `inpaint_local_enabled=True`. VACE slices to mask tokens; DiT receives hints only at masked positions. Loss is MSE on masked latent positions only.

**Stage 2.** Same data flow plus `downsampled_input_latents` produced by `WanVideoUnit_InpaintMask`. `model_fn_wan_video` runs with both flags. `gp = dit.global_patch_embedding(downsampled_input_latents)` is concatenated into `context`; cross-attn `k_img`/`v_img` consume the image-context half. Loss unchanged. Stage-2 freeze pass ensures only `global_patch_embedding` + cross-attn proj subsets receive gradients.

## Testing strategy

(See brainstorm Section 5 for details.)

- Unit: `compute_downsampled_masks_tensor` correctness, VACE mask-slicing forward shape and scatter-back behavior.
- Integration (CPU, tiny configs): one forward + backward step in each stage; assert correct `requires_grad` subsets.
- Upstream regression: bit-identical outputs of `model_fn_wan_video` / `VaceWanModel.forward` with all flags off vs a pre-change snapshot.
- Manual GPU: 100-step run of each stage; inference script produces a video.

## Acceptance criteria

1. Unit + integration + regression tests pass.
2. Upstream Wan-VACE training/inference scripts (`Wan2.1-VACE-1.3B.sh`, `Wan2.1-VACE-14B.sh`) run unchanged.
3. Stage 1 + Stage 2 scripts run end-to-end on user's machine for ≥100 steps, saving valid checkpoints.
4. Stage 2 checkpoint diff vs base contains only `global_patch_embedding.*` and `blocks.*.cross_attn.{k_img,k_img_norm,v_img,zero_proj}.*`.
5. Inference script produces a video where the masked region differs visibly from the masked input.

## Out of scope

- Wan 2.2 VACE (`vace2`/`dit2` MoE).
- DAVIS / VPBench evaluation harness.
- The fork's bbox-controller branch (dead code — not ported).
- Decoder LoRA stage (the fork's optional Stage 3).
- The fork's mask dilation flags (`dilate_0` / `dilate_1`). The `dilate_mask` helper is ported but not threaded into a CLI flag in this round.
- Sliding-window / long-video inference variations.

## Open implementation questions (resolve during plan-writing, not blocking spec approval)

- Exact `scale_factor` for `downsampled_input_latents` — fork uses 0.5; verify by reading fork's effective `f_down`/`h_down`/`w_down` shapes against the cross-attn context dim. Document choice in the plan.
- Whether `WanVideoUnit_InpaintMask` should also emit `downsampled_vace_mask` for inference latent surgery, or whether the inference path derives that separately. Likely the former — keeps surgery code self-contained.
- Whether `LoadVideoMaskPair` is a separate operator or a thin wrapper over two `LoadVideo` invocations with shared frame indexing.
