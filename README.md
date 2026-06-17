# <p align="center"> EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing </p>

#####  <p align="center"> [Yehonathan Litman](https://yehonathanlitman.github.io/), [Shikun Liu](https://shikun.io), [Dario Seyb](https://darioseyb.com/), [Nicholas Milef](https://www.nicholasmilef.com/), [Yang Zhou](https://linktr.ee/mangosister), [Carl Marshall](https://scholar.google.com/citations?user=xWD7ZRkAAAAJ), [Shubham Tulsiani](https://www.cs.cmu.edu/~stulsian/), [Caleb Leak](https://www.linkedin.com/in/calebleak)</p>
##### <p align="center"> CVPR 2026

#### <p align="center">[📑 Paper](http://arxiv.org/abs/2602.15031) | [🖥️ Webpage](https://yehonathanlitman.github.io/edit_ctrl/) | [🤗 Weights](https://huggingface.co/thebluser) <br><br>

<img width="1379" height="847" alt="EditCtrl teaser" src="https://github.com/user-attachments/assets/65c6faba-ad8b-4ac6-9efa-f6179328bb20" />

<br>

> **Disclaimer:** this is a public reimplementation of EditCtrl built on top of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and trained on the [VideoPainter](https://github.com/TencentARC/VideoPainter) VPData dataset. It reproduces the local + global control design described in the paper across Wan 2.1 VACE (1.3B and 14B) and Wan 2.2 VACE Fun A14B.

## Installation

```bash
git clone https://github.com/yehonathanlitman/diffsynth-studio-editctrl.git
cd diffsynth-studio-editctrl

conda create -n editctrl python=3.11 -y
conda activate editctrl
pip install -e .
pip install huggingface_hub
```

Tested on CUDA 12 with an H100 card.

## Quick start

Five sample `(video, mask)` pairs are shipped in `examples/wanvideo/model_inference/samples/`. The inference scripts download the EditCtrl weights from Hugging Face on first run.

| Model | HF repo |
|---|---|
| Wan 2.1 VACE 1.3B | [`thebluser/Wan2.1-VACE-1.3B-editctrl`](https://huggingface.co/thebluser/Wan2.1-VACE-1.3B-editctrl) |
| Wan 2.1 VACE 14B | [`thebluser/Wan2.1-VACE-14B-editctrl`](https://huggingface.co/thebluser/Wan2.1-VACE-14B-editctrl) |
| Wan 2.2 VACE Fun A14B | [`thebluser/Wan2.2-VACE-Fun-A14B-editctrl`](https://huggingface.co/thebluser/Wan2.2-VACE-Fun-A14B-editctrl) |

> **Note:** these checkpoints were trained with less compute resources than the internal version, so quality may be subpar.

**Wan 2.1 VACE 1.3B (smallest, fastest):**
```bash
python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_editctrl.py \
  --input_video examples/wanvideo/model_inference/samples/sample_00_video.mp4 \
  --input_mask  examples/wanvideo/model_inference/samples/sample_00_mask.mp4 \
  --prompt "$(jq -r .sample_00 examples/wanvideo/model_inference/samples/prompts.json)" \
  --output_path sample_00_out.mp4
```

**Wan 2.1 VACE 14B:**
```bash
python examples/wanvideo/model_inference/Wan2.1-VACE-14B_editctrl.py \
  --input_video examples/wanvideo/model_inference/samples/sample_00_video.mp4 \
  --input_mask  examples/wanvideo/model_inference/samples/sample_00_mask.mp4 \
  --prompt "..." \
  --output_path sample_00_out.mp4
```

**Wan 2.2 VACE Fun A14B (MoE high/low-noise experts):**
```bash
python examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B_editctrl.py \
  --input_video examples/wanvideo/model_inference/samples/sample_00_video.mp4 \
  --input_mask  examples/wanvideo/model_inference/samples/sample_00_mask.mp4 \
  --prompt "..." \
  --output_path sample_00_out.mp4
```

> **Wan 2.2 note:** the A14B script defaults to **local-only** inference. Excluding the global EditCtrl DiT weights proved more stable on the MoE experts. Pass `--enable_global` to opt back into the global path.

See `examples/wanvideo/model_inference/samples/README.md` for the full text prompts that go with each sample clip.

## Training

### Dataset — VPData from VideoPainter

The public-release editctrl weights were trained on the [VPData](https://huggingface.co/datasets/TencentARC/VideoPainter) dataset released with the [VideoPainter](https://github.com/TencentARC/VideoPainter) paper. To train your own checkpoint:

1. Clone the VPData dataset from [Hugging Face](https://huggingface.co/datasets/TencentARC/VideoPainter) into `VPData/` at the repo root. This brings the train / val / test CSVs, the videovo raw video clips, and all per-clip mask `.npz` files.
2. Run `python download_vpdata.py` from the repo root. It reads `VPData/pexels.csv` and downloads the raw Pexels clips that the CSVs reference.

### Base model weights

The training scripts expect Wan base weights cached under `models/`:
- `models/Wan-AI/Wan2.1-VACE-1.3B/`
- `models/Wan-AI/Wan2.1-VACE-14B/`
- `models/PAI/Wan2.2-VACE-Fun-A14B/`

These are pulled lazily from ModelScope the first time you run training or inference which can be slow. To download them faster:

```bash
pip install -U huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B            --local-dir models/Wan-AI/Wan2.1-VACE-1.3B
huggingface-cli download Wan-AI/Wan2.1-VACE-14B             --local-dir models/Wan-AI/Wan2.1-VACE-14B
huggingface-cli download alibaba-pai/Wan2.2-VACE-Fun-A14B   --local-dir models/PAI/Wan2.2-VACE-Fun-A14B
```

### Train

Each script trains the **local** (VACE LoRA) and **global** (DiT delta) stages back-to-back, capped at 10,000 iterations per stage:

```bash
# Wan 2.1 VACE 1.3B
bash examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B_inpaint_editctrl_vpdata.sh

# Wan 2.1 VACE 14B
bash examples/wanvideo/model_training/full/Wan2.1-VACE-14B_inpaint_editctrl_vpdata.sh

# Wan 2.2 VACE Fun A14B — high-noise expert
bash examples/wanvideo/model_training/full/Wan2.2-VACE-Fun-A14B_inpaint_editctrl_vpdata_high_noise.sh

# Wan 2.2 VACE Fun A14B — low-noise expert
bash examples/wanvideo/model_training/full/Wan2.2-VACE-Fun-A14B_inpaint_editctrl_vpdata_low_noise.sh
```

Checkpoints are written to `./models/train/<base>_editctrl_local[_noise]/step-NNNNN.safetensors` and `./models/train/<base>_editctrl_global[_noise]/step-NNNNN.safetensors`. The global stage automatically picks up `step-10000.safetensors` from the local stage.

For inference on VPData, run the `_vpdata` variants of the inference scripts (`examples/wanvideo/model_inference/*_editctrl_vpdata.py`).

### Tips for Quality

- **EditCtrl was trained on detailed description-based text prompts, not simple instruction-based text prommpts for editing.** For example, if your video contains a room with a table and you would like to add a cup, do not use a prompt like `Add a cup`, instead use `A glass cup resting on a table in an old English home, its delicately carved details shimmering in the sun.`
- EditCtrl was trained with mask augmentations but semantic masks like ones inferred with SAM3 lead to better results.
- Videos with high motion and blur lead to artifacts in the edits due to the weakness of the Wan VAE.

## Citation

```bibtex
@inproceedings{litman2026editctrl,
    title={EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing},
    author={Litman, Yehonathan and Liu, Shikun and Seyb, Dario and Milef, Nicholas and Zhou, Yang and Marshall, Carl and Tulsiani, Shubham and Leak, Caleb},
    booktitle={CVPR},
    year={2026}
}
```

## Acknowledgements

EditCtrl is built on top of incredible open-source work:
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — the training/inference framework this repo extends.
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and [Wan 2.2](https://github.com/Wan-Video/Wan2.2) — the base VACE video diffusion models.
- [VideoPainter](https://github.com/TencentARC/VideoPainter) — for the VPData inpainting dataset.
