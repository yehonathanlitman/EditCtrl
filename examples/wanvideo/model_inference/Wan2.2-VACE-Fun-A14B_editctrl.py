"""editctrl inpainting inference on Wan2.2-VACE-Fun-A14B (MoE: high/low noise experts).

Default behavior: LOCAL-ONLY inference (LoRA on pipe.vace and pipe.vace2).
Excluding the global editctrl DiT weights for Wan 2.2 proved more stable
in our testing. Pass --enable_global to opt back into the global path.

Checkpoints default to downloads from Hugging Face
(thebluser/Wan2.2-VACE-Fun-A14B-editctrl). Pass any of
--local_ckpt_high / --local_ckpt_low / --global_ckpt_high / --global_ckpt_low
to override with a local file path.

Usage:
    python examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B_editctrl.py \\
        --input_video path/to/input.mp4 \\
        --input_mask  path/to/mask.mp4 \\
        --prompt "a cat sitting on a chair" \\
        --output_path inpainted.mp4
"""
import argparse
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from diffsynth.core.loader.file import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

HF_REPO = "thebluser/Wan2.2-VACE-Fun-A14B-editctrl"
HF_FILES = {
    "local_high":  "local_high_noise.safetensors",
    "local_low":   "local_low_noise.safetensors",
    "global_high": "global_high_noise.safetensors",
    "global_low":  "global_low_noise.safetensors",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", required=True)
    p.add_argument("--input_mask",  required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output_path", default="editctrl_out.mp4")
    p.add_argument("--enable_global", action="store_true",
                   help="Enable the global editctrl DiT path. Default off — local-only inference proved more stable for Wan 2.2 in our testing.")
    p.add_argument("--local_ckpt_high",  default=None,
                   help=f"Local file path. Default: download {HF_REPO}/{HF_FILES['local_high']}")
    p.add_argument("--local_ckpt_low",   default=None,
                   help=f"Local file path. Default: download {HF_REPO}/{HF_FILES['local_low']}")
    p.add_argument("--global_ckpt_high", default=None,
                   help=f"Only used when --enable_global is set. Default: download {HF_REPO}/{HF_FILES['global_high']}")
    p.add_argument("--global_ckpt_low",  default=None,
                   help=f"Only used when --enable_global is set. Default: download {HF_REPO}/{HF_FILES['global_low']}")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width",  type=int, default=720)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def _zero_inside_mask(video_frames, mask_frames):
    zeroed = []
    for v, m in zip(video_frames, mask_frames):
        v_arr = np.array(v).copy()
        m_arr = np.array(m)
        if m_arr.ndim == 3:
            m_arr = m_arr.mean(axis=2)
        v_arr[m_arr > 127] = 0
        zeroed.append(Image.fromarray(v_arr))
    return zeroed


def main():
    args = parse_args()
    local_high = args.local_ckpt_high or hf_hub_download(repo_id=HF_REPO, filename=HF_FILES["local_high"])
    local_low  = args.local_ckpt_low  or hf_hub_download(repo_id=HF_REPO, filename=HF_FILES["local_low"])
    print(f"local high LoRA  : {local_high}")
    print(f"local low LoRA   : {local_low}")
    if args.enable_global:
        global_high = args.global_ckpt_high or hf_hub_download(repo_id=HF_REPO, filename=HF_FILES["global_high"])
        global_low  = args.global_ckpt_low  or hf_hub_download(repo_id=HF_REPO, filename=HF_FILES["global_low"])
        print(f"global high delta: {global_high}")
        print(f"global low delta : {global_low}")
    else:
        print("global path disabled (pass --enable_global to enable)")

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
        enable_inpaint_global=args.enable_global,
    )

    pipe.load_lora(pipe.vace, local_high)
    print(f"loaded local high LoRA into pipe.vace")
    if getattr(pipe, "vace2", None) is not None:
        pipe.load_lora(pipe.vace2, local_low)
        print(f"loaded local low LoRA into pipe.vace2")

    if args.enable_global:
        sd_high = load_state_dict(global_high)
        _, unexpected = pipe.dit.load_state_dict(sd_high, strict=False)
        print(f"global high delta: {len(sd_high)} keys, {len(unexpected)} unexpected")
        if getattr(pipe, "dit2", None) is not None:
            sd_low = load_state_dict(global_low)
            _, unexpected = pipe.dit2.load_state_dict(sd_low, strict=False)
            print(f"global low delta : {len(sd_low)} keys, {len(unexpected)} unexpected")

    video_frames = VideoData(args.input_video, height=args.height, width=args.width).raw_data()[: args.num_frames]
    mask_frames  = VideoData(args.input_mask,  height=args.height, width=args.width).raw_data()[: args.num_frames]
    vace_video_frames = _zero_inside_mask(video_frames, mask_frames)

    out = pipe(
        prompt=args.prompt,
        input_video=video_frames, denoising_strength=1.0,
        vace_video=vace_video_frames, vace_video_mask=mask_frames,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        inpaint_local_enabled=True,
        inpaint_global_enabled=args.enable_global,
        inpaint_latent_surgery=True,
    )
    save_video(out, args.output_path, fps=args.fps)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
