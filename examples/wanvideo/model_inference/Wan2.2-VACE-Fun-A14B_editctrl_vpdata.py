import argparse
import os
import torch
from diffsynth.core.loader.file import load_state_dict
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_ckpt_high", default="./models/train/Wan2.2-VACE-Fun-A14B_editctrl_local_high_noise/step-10000.safetensors")
    p.add_argument("--local_ckpt_low",  default="./models/train/Wan2.2-VACE-Fun-A14B_editctrl_local_low_noise/step-10000.safetensors")
    p.add_argument("--global_ckpt_high", default="./models/train/Wan2.2-VACE-Fun-A14B_editctrl_global_high_noise/step-10000.safetensors")
    p.add_argument("--global_ckpt_low",  default="./models/train/Wan2.2-VACE-Fun-A14B_editctrl_global_low_noise/step-10000.safetensors")
    p.add_argument("--val_csv",          default="VPData/pexels_videovo_val_dataset.csv")
    p.add_argument("--base_path",        default="VPData/videovo_raw_videos")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height",     type=int, default=480)
    p.add_argument("--width",      type=int, default=720)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--dataset_fps", type=int, default=8)
    p.add_argument("--output_dir", default="outputs/editctrl_2.2_A14B_test")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files=False,
        enable_inpaint_global=True,
    )

    pipe.load_lora(pipe.vace, args.local_ckpt_high)
    print(f"Loaded high-noise local LoRA into pipe.vace from {args.local_ckpt_high}")
    if getattr(pipe, "vace2", None) is not None:
        pipe.load_lora(pipe.vace2, args.local_ckpt_low)
        print(f"Loaded low-noise local LoRA into pipe.vace2 from {args.local_ckpt_low}")

    sd_high = load_state_dict(args.global_ckpt_high)
    _, unexpected = pipe.dit.load_state_dict(sd_high, strict=False)
    print(f"Loaded high-noise global DiT delta into pipe.dit from {args.global_ckpt_high}: {len(sd_high)} keys, {len(unexpected)} unexpected")
    if getattr(pipe, "dit2", None) is not None:
        sd_low = load_state_dict(args.global_ckpt_low)
        _, unexpected = pipe.dit2.load_state_dict(sd_low, strict=False)
        print(f"Loaded low-noise global DiT delta into pipe.dit2 from {args.global_ckpt_low}: {len(sd_low)} keys, {len(unexpected)} unexpected")

    ds = VideoInpaintingDataset(
        base_path=args.base_path,
        metadata_path=args.val_csv,
        height=args.height, width=args.width, num_frames=args.num_frames,
        fps=args.dataset_fps, is_train=False, repeat=1,
    )
    print(f"Dataset: {len(ds)} val rows; reading first {args.n_samples}")

    import numpy as np
    from PIL import Image as _PILImage

    def _zero_inside_mask(video_frames, mask_frames):
        zeroed = []
        for v, m in zip(video_frames, mask_frames):
            v_arr = np.array(v).copy()
            m_arr = np.array(m)
            if m_arr.ndim == 3:
                m_arr = m_arr.mean(axis=2)
            inside = m_arr > 127
            v_arr[inside] = 0
            zeroed.append(_PILImage.fromarray(v_arr))
        return zeroed

    saved = []
    for i in range(args.n_samples):
        sample = ds[i]
        prompt = sample["prompt"]
        video_frames = sample["video"]
        mask_frames = sample["vace_video_mask"]
        vace_video_frames = _zero_inside_mask(video_frames, mask_frames)

        out = pipe(
            prompt=prompt,
            input_video=video_frames, denoising_strength=1.0,
            vace_video=vace_video_frames, vace_video_mask=mask_frames,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            inpaint_local_enabled=True,
            inpaint_global_enabled=True,
            inpaint_latent_surgery=True,
        )

        in_path  = os.path.abspath(os.path.join(args.output_dir, f"sample_{i:02d}_input.mp4"))
        mk_path  = os.path.abspath(os.path.join(args.output_dir, f"sample_{i:02d}_mask.mp4"))
        out_path = os.path.abspath(os.path.join(args.output_dir, f"sample_{i:02d}_output.mp4"))
        save_video(video_frames, in_path, fps=args.fps)
        save_video(mask_frames, mk_path, fps=args.fps)
        save_video(out, out_path, fps=args.fps)
        saved.append((in_path, mk_path, out_path, prompt[:100]))
        print(f"[{i+1}/{args.n_samples}] Saved input/mask/output for sample {i:02d}")

    print("\n=== Saved videos ===")
    for in_p, mk_p, out_p, caption in saved:
        print(f"input : {in_p}")
        print(f"mask  : {mk_p}")
        print(f"output: {out_p}")
        print(f"prompt: {caption}...")
        print()


if __name__ == "__main__":
    main()
