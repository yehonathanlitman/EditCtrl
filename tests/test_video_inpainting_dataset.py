import os
import pytest
from pathlib import Path

VPDATA_ROOT = Path(os.environ.get("VPDATA_ROOT", "VPData"))


@pytest.mark.skipif(not VPDATA_ROOT.exists(), reason="VPData not present on this machine")
def test_video_inpainting_dataset_loads_one_videovo_sample():
    from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset

    ds = VideoInpaintingDataset(
        base_path=str(VPDATA_ROOT / "videovo_raw_videos"),
        metadata_path=str(VPDATA_ROOT / "pexels_videovo_train_dataset.csv"),
        height=480,
        width=720,
        num_frames=17,
        fps=8,
        is_train=True,
        repeat=1,
    )

    assert len(ds) > 0

    metas = ds.metas
    videovo_idx = None
    for i, row in enumerate(metas):
        if ".0.mp4" in str(row[0]):
            videovo_idx = i
            break
    assert videovo_idx is not None, "No videovo rows found in filtered dataset"

    sample = ds[videovo_idx]
    assert set(sample.keys()) >= {"prompt", "video", "vace_video", "vace_video_mask"}

    assert isinstance(sample["prompt"], str)
    assert len(sample["video"]) == 17
    assert len(sample["vace_video"]) == 17
    assert len(sample["vace_video_mask"]) == 17

    assert sample["vace_video_mask"][0].mode == "RGB"
    assert sample["video"][0].mode == "RGB"
    assert sample["video"][0].size == (720, 480)
    assert sample["vace_video_mask"][0].size == (720, 480)
    assert sample["vace_video"][0].mode == "RGB"
    assert sample["vace_video"][0].size == (720, 480)


@pytest.mark.skipif(not VPDATA_ROOT.exists(), reason="VPData not present on this machine")
def test_video_inpainting_dataset_loads_one_pexels_sample():
    from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset

    ds = VideoInpaintingDataset(
        base_path=str(VPDATA_ROOT / "videovo_raw_videos"),
        metadata_path=str(VPDATA_ROOT / "pexels_videovo_train_dataset.csv"),
        height=480,
        width=720,
        num_frames=17,
        fps=8,
        is_train=True,
        repeat=1,
    )

    metas = ds.metas
    pexels_idx = None
    for i, row in enumerate(metas):
        if ".0.mp4" not in str(row[0]) and ".mp4" in str(row[0]):
            pexels_idx = i
            break
    assert pexels_idx is not None, "No pexels rows found in filtered dataset"

    sample = ds[pexels_idx]
    assert set(sample.keys()) >= {"prompt", "video", "vace_video", "vace_video_mask"}
    assert isinstance(sample["prompt"], str)
    assert len(sample["video"]) == 17
    assert len(sample["vace_video_mask"]) == 17
    assert sample["vace_video_mask"][0].mode == "RGB"
    assert sample["video"][0].mode == "RGB"
    assert sample["video"][0].size == (720, 480)
    assert sample["vace_video_mask"][0].size == (720, 480)


@pytest.mark.skipif(not VPDATA_ROOT.exists(), reason="VPData not present on this machine")
def test_video_inpainting_dataset_repeat_multiplies_length():
    from diffsynth.core.data.video_inpainting_dataset import VideoInpaintingDataset

    ds1 = VideoInpaintingDataset(
        base_path=str(VPDATA_ROOT / "videovo_raw_videos"),
        metadata_path=str(VPDATA_ROOT / "pexels_videovo_train_dataset.csv"),
        height=480,
        width=720,
        num_frames=17,
        fps=8,
        repeat=1,
    )
    ds3 = VideoInpaintingDataset(
        base_path=str(VPDATA_ROOT / "videovo_raw_videos"),
        metadata_path=str(VPDATA_ROOT / "pexels_videovo_train_dataset.csv"),
        height=480,
        width=720,
        num_frames=17,
        fps=8,
        repeat=3,
    )
    assert len(ds3) == 3 * len(ds1)
