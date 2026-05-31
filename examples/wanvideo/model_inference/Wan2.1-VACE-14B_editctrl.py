"""editctrl inpainting inference on Wan2.1-VACE-14B.

Default behavior: download the local LoRA and global DiT delta from
Hugging Face (thebluser/Wan2.1-VACE-14B-editctrl). Pass --local_ckpt
or --global_ckpt to override with a local file path.

Usage:
    python examples/wanvideo/model_inference/Wan2.1-VACE-14B_editctrl.py \\
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

HF_REPO = "thebluser/Wan2.1-VACE-14B-editctrl"
HF_LOCAL_FILE = "local.safetensors"
HF_GLOBAL_FILE = "global.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", required=True)
    p.add_argument("--input_mask",  required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output_path", default="editctrl_out.mp4")
    p.add_argument("--local_ckpt",  default=None,
                   help=f"Local file path for the local LoRA. Default: download {HF_REPO}/{HF_LOCAL_FILE}")
    p.add_argument("--global_ckpt", default=None,
                   help=f"Local file path for the global DiT delta. Default: download {HF_REPO}/{HF_GLOBAL_FILE}")
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
    local_ckpt  = args.local_ckpt  or hf_hub_download(repo_id=HF_REPO, filename=HF_LOCAL_FILE)
    global_ckpt = args.global_ckpt or hf_hub_download(repo_id=HF_REPO, filename=HF_GLOBAL_FILE)
    print(f"local LoRA       : {local_ckpt}")
    print(f"global DiT delta : {global_ckpt}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        redirect_common_files=False,
        enable_inpaint_global=True,
    )
    pipe.load_lora(pipe.vace, local_ckpt)
    sd = load_state_dict(global_ckpt)
    _, unexpected = pipe.dit.load_state_dict(sd, strict=False)
    print(f"global delta: {len(sd)} keys loaded, {len(unexpected)} unexpected")

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
        inpaint_global_enabled=True,
        inpaint_latent_surgery=True,
    )
    save_video(out, args.output_path, fps=args.fps)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
