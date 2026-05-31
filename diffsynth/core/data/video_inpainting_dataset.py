from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import os
import random


class VideoInpaintingDataset(Dataset):

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        height: int,
        width: int,
        num_frames: int,
        fps: int = 8,
        is_train: bool = True,
        repeat: int = 1,
    ):
        super().__init__()
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.csv_dir = str(Path(metadata_path).parent)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.fps = fps
        self.is_train = is_train
        self.repeat = max(1, repeat)
        self.load_from_cache = False

        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
        if not Path(base_path).exists():
            raise FileNotFoundError(f"base_path not found: {base_path}")

        self.metas = self._load_metas()

    def _load_metas(self) -> np.ndarray:
        df = pd.read_csv(self.metadata_path)
        df = df[df["caption"].str.len() > 50]
        df = df[(df["end_frame"] - df["start_frame"]) // (df["fps"] // self.fps) >= self.num_frames]
        return df.values

    def _resolve_video_path(self, row_path: str) -> str:
        if ".0.mp4" in row_path:
            prefix = row_path[:9]
            return os.path.join(self.base_path, prefix, row_path)
        else:
            pexels_base = self.base_path.replace("videovo_raw_videos", "pexels/pexels/raw_video")
            prefix = row_path[:9]
            return os.path.join(pexels_base, prefix, row_path)

    def _resolve_mask_path(self, row_path: str) -> str:
        stem = row_path.split(".")[0]
        if ".0.mp4" in row_path:
            return os.path.join(self.csv_dir, "videovo_masks", stem, "all_masks.npz")
        else:
            return os.path.join(self.csv_dir, "pexels_masks", stem, "all_masks.npz")

    def __len__(self) -> int:
        return len(self.metas) * self.repeat

    def __getitem__(self, index: int) -> dict:
        n = len(self.metas)
        last_error: Optional[BaseException] = None
        for _ in range(n):
            try:
                row = self.metas[index % n]
                row_path = row[0]
                video_path = self._resolve_video_path(row_path)
                mask_path = self._resolve_mask_path(row_path)
                if not (os.path.exists(video_path) and os.path.exists(mask_path)):
                    index = random.randint(0, n - 1)
                    continue

                _, start_frame, end_frame, source_fps, mask_id, caption = row
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                source_fps = int(source_fps)
                mask_id = int(mask_id)
                step = max(1, source_fps // self.fps)

                reader = imageio.get_reader(video_path)
                try:
                    frames = []
                    for frame_id in range(start_frame, end_frame, step):
                        frames.append(reader.get_data(frame_id))
                finally:
                    reader.close()
                frames = frames[: self.num_frames]

                all_masks = np.load(mask_path)["arr_0"][start_frame:end_frame]
                binary_masks = (all_masks == mask_id).astype(np.uint8) * 255
                binary_masks = binary_masks[::step][: self.num_frames]

                target_size = (self.width, self.height)
                video_pil = [
                    Image.fromarray(f).resize(target_size, Image.BILINEAR)
                    for f in frames
                ]
                mask_pil = [
                    Image.fromarray(m, mode="L").convert("RGB").resize(target_size, Image.NEAREST)
                    for m in binary_masks
                ]

                return {
                    "prompt": str(caption),
                    "video": video_pil,
                    "vace_video": video_pil,
                    "vace_video_mask": mask_pil,
                }
            except Exception as e:
                last_error = e
                print(f"[VideoInpaintingDataset] skipping row index={index % n} due to load error: {type(e).__name__}: {e}")
                index = random.randint(0, n - 1)

        raise RuntimeError(
            f"Failed to load any row from {self.metadata_path!r} after {n} attempts; "
            f"last error: {type(last_error).__name__}: {last_error}"
        )
