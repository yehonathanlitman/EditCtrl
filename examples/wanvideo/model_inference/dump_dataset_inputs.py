import argparse
import os
import numpy as np
from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset
from diffsynth.utils.data import save_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv",   default="VPData/pexels_videovo_val_dataset.csv")
    p.add_argument("--base_path", default="VPData/videovo_raw_videos")
    p.add_argument("--n_samples", type=int, default=2)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width",  type=int, default=720)
    p.add_argument("--dataset_fps", type=int, default=8)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--output_dir", default="outputs/local_step20_test")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ds = VideoInpaintingDataset(
        base_path=args.base_path,
        metadata_path=args.val_csv,
        height=args.height, width=args.width, num_frames=args.num_frames,
        fps=args.dataset_fps, is_train=False, repeat=1,
    )
    for i in range(args.n_samples):
        sample = ds[i]
        in_path = os.path.abspath(os.path.join(args.output_dir, f"sample_{i:02d}_input.mp4"))
        mk_path = os.path.abspath(os.path.join(args.output_dir, f"sample_{i:02d}_mask.mp4"))
        save_video(sample["video"], in_path, fps=args.fps)
        save_video(sample["vace_video_mask"], mk_path, fps=args.fps)
        print(f"sample_{i:02d}: {in_path}")
        print(f"sample_{i:02d}: {mk_path}")
        print(f"sample_{i:02d}: prompt={sample['prompt'][:100]}...")


if __name__ == "__main__":
    main()
