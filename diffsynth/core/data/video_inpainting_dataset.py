from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import cv2
import math
import os
import random


# Mask augmentations ported from DiffSynthInpaint/diffsynth/trainers/video_inpainting_dataset.py.
# Per-clip choice of {brush, rect, ellipse, circle, random_brush} replacing the
# raw segmentation mask with a shape-randomized version, so training generalizes
# beyond the exact mask geometries in VPData. Applied with probability
# `mask_aug_prob` per clip when `is_train=True`.

def _generate_random_brush(h, w):
    mask = Image.new('L', (w, h), 0)
    average_radius = math.sqrt(h * h + w * w) / 8
    max_tries = 5
    min_num_vertex, max_num_vertex = 1, 8
    mean_angle = 0.0
    angle_range = 0.01
    min_width, max_width = 128, 256

    num_tries = np.random.choice(list(range(max_tries)), p=[0.05, 0.3, 0.3, 0.3, 0.05])
    for _ in range(num_tries):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles, vertex = [], []
        for i in range(num_vertex):
            angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max)
                          if i % 2 == 0 else np.random.uniform(angle_min, angle_max))
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2),
                        0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2), fill=1)

    arr = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        arr = np.flip(arr, 0)
    if np.random.random() > 0.5:
        arr = np.flip(arr, 1)
    return arr


def _transform_video_masks(
    video_masks,                # (F, H, W, 1) uint8 in {0, 1}
    p_brush=0.3, p_rect=0.3, p_ellipse=0.2, p_circle=0.2, p_random_brush=0.0,
    margin_ratio=0.1, shape_scale_min=1.1, shape_scale_max=1.5, brush_iterations=1,
):
    """Per-clip mask shape randomization. Same parameters for every frame of a clip."""
    F, H, W, C = video_masks.shape
    out = np.zeros_like(video_masks)

    choice = np.random.choice(
        ['brush', 'rect', 'ellipse', 'circle', 'random_brush'],
        p=[p_brush, p_rect, p_ellipse, p_circle, p_random_brush],
    )

    if choice == 'brush':
        morph_type = np.random.choice(['dilate_erode', 'erode_dilate', 'dilate_only', 'combined'])
        use_blur = np.random.random() < 0.1
    elif choice == 'random_brush':
        first_frame_brush = _generate_random_brush(H, W)
    elif choice == 'rect':
        rect_angle = 0
        width_scale = np.random.uniform(shape_scale_min, shape_scale_max)
        height_scale = np.random.uniform(shape_scale_min, shape_scale_max)
    elif choice == 'ellipse':
        width_scale = np.random.uniform(shape_scale_min / 2, shape_scale_max / 2)
        height_scale = np.random.uniform(shape_scale_min / 2, shape_scale_max / 2)
        angle = np.random.uniform(0, 360)
    else:  # circle
        radius_scale = np.random.uniform(shape_scale_min / 2, shape_scale_max / 2)

    if choice in ('rect', 'ellipse', 'circle'):
        first_frame = video_masks[0]
        y_idx, x_idx = np.where(first_frame[:, :, 0] > 0)
        if len(y_idx) == 0 or len(x_idx) == 0:
            return video_masks
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        margin = int(min(H, W) * margin_ratio)
        x_min = max(0, x_min - np.random.randint(0, margin + 1))
        x_max = min(W, x_max + np.random.randint(0, margin + 1))
        y_min = max(0, y_min - np.random.randint(0, margin + 1))
        y_max = min(H, y_max + np.random.randint(0, margin + 1))
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        w_box, h_box = x_max - x_min, y_max - y_min

        first_frame_shape = np.zeros((H, W), dtype=np.uint8)
        if choice == 'rect':
            rect = ((float(cx), float(cy)),
                    (float(w_box * width_scale), float(h_box * height_scale)),
                    float(rect_angle))
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(first_frame_shape, [box], 1)
        elif choice == 'ellipse':
            axes = (int(w_box * width_scale), int(h_box * height_scale))
            cv2.ellipse(first_frame_shape, (cx, cy), axes, angle, 0, 360, 1, -1)
        else:
            radius = int(max(w_box, h_box) * radius_scale)
            cv2.circle(first_frame_shape, (cx, cy), radius, 1, -1)

    kernel = np.ones((32, 32), np.uint8)
    for f in range(F):
        src = video_masks[f, :, :, 0].astype(np.uint8)
        if choice == 'random_brush':
            out[f, :, :, 0] = first_frame_brush
        elif choice in ('rect', 'ellipse', 'circle'):
            out[f, :, :, 0] = first_frame_shape
        else:  # brush morph
            if morph_type == 'dilate_erode':
                m = cv2.erode(cv2.dilate(src, kernel, iterations=brush_iterations), kernel, iterations=brush_iterations)
            elif morph_type == 'erode_dilate':
                m = cv2.dilate(cv2.erode(src, kernel, iterations=brush_iterations), kernel, iterations=brush_iterations)
            elif morph_type == 'dilate_only':
                m = cv2.dilate(src, kernel, iterations=brush_iterations)
            else:  # combined
                eroded = cv2.erode(src, kernel, iterations=brush_iterations)
                opened = cv2.dilate(eroded, kernel, iterations=brush_iterations)
                dilated = cv2.dilate(opened, kernel, iterations=brush_iterations)
                m = cv2.erode(dilated, kernel, iterations=brush_iterations)
            if use_blur:
                m = cv2.GaussianBlur(m, (3, 3), 0)
                m = (m > 0.5).astype(np.uint8)
            out[f, :, :, 0] = m
        if C > 1:
            out[f, :, :, 1:] = out[f, :, :, 0:1]
    return out


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
        # Mask augmentation params (ported from DiffSynthInpaint fork).
        # Probability of applying ANY augmentation per clip during training.
        mask_aug_prob: float = 0.7,
        # Per-choice probabilities (must sum to 1).
        p_brush: float = 0.3,
        p_rect: float = 0.3,
        p_ellipse: float = 0.2,
        p_circle: float = 0.2,
        p_random_brush: float = 0.0,
        margin_ratio: float = 0.1,
        shape_scale_min: float = 1.1,
        shape_scale_max: float = 1.5,
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
        self.mask_aug_prob = mask_aug_prob
        self.mask_aug_kwargs = dict(
            p_brush=p_brush, p_rect=p_rect, p_ellipse=p_ellipse,
            p_circle=p_circle, p_random_brush=p_random_brush,
            margin_ratio=margin_ratio,
            shape_scale_min=shape_scale_min, shape_scale_max=shape_scale_max,
        )

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
                binary_masks = (all_masks == mask_id).astype(np.uint8)  # (F, H, W) in {0, 1}
                binary_masks = binary_masks[::step][: self.num_frames]

                if self.is_train and self.mask_aug_prob > 0 and random.random() < self.mask_aug_prob:
                    aug_in = binary_masks[..., np.newaxis]  # (F, H, W, 1)
                    binary_masks = _transform_video_masks(aug_in, **self.mask_aug_kwargs)[..., 0]

                binary_masks = binary_masks * 255  # back to {0, 255} for PIL

                target_size = (self.width, self.height)
                video_pil = [
                    Image.fromarray(f).resize(target_size, Image.BILINEAR)
                    for f in frames
                ]
                mask_pil = [
                    Image.fromarray(m, mode="L").convert("RGB").resize(target_size, Image.NEAREST)
                    for m in binary_masks
                ]

                # Zero the inside-mask region of vace_video so `reactive` is 0.
                # Matches the _zero_inside_mask step the *_editctrl.py scripts do
                # at inference; without it, VACE's reactive channel would leak
                # the GT pixels inside the mask during training.
                vace_video_pil = []
                for v, m in zip(video_pil, mask_pil):
                    v_arr = np.array(v).copy()
                    m_arr = np.array(m)
                    if m_arr.ndim == 3:
                        m_arr = m_arr.mean(axis=2)
                    v_arr[m_arr > 127] = 0
                    vace_video_pil.append(Image.fromarray(v_arr))

                return {
                    "prompt": str(caption),
                    "video": video_pil,
                    "vace_video": vace_video_pil,
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
