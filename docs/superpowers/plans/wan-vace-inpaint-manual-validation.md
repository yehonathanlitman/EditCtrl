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
