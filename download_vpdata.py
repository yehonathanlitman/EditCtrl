import pandas as pd
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Helper: Extract video IDs from path column ===
def extract_valid_video_ids(dataset_paths):
    video_ids = set()
    for path in dataset_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for p in df["path"]:
            try:
                # Extract the number after the underscore
                vid_id = p.split("_")[1].replace(".mp4", "")
                video_ids.add(vid_id)
            except Exception:
                pass  # Skip malformed entries
    return video_ids

# === Step 1: Get all valid video IDs from train/val/test datasets ===
dataset_files = [
    "VPData/pexels_videovo_train_dataset.csv",
    "VPData/pexels_videovo_test_dataset.csv",
    "VPData/pexels_videovo_val_dataset.csv"
]

valid_video_ids = extract_valid_video_ids(dataset_files)

# === Step 2: Load original video metadata ===
pexels_df = pd.read_csv("VPData/pexels.csv")

# === Step 3: Setup download directory ===
output_dir = "VPData"
base_path = os.path.join(output_dir, "pexels/pexels/raw_video")
os.makedirs(base_path, exist_ok=True)

# === Step 4: Define downloader ===
def download_video(row):
    index, video_url, video_id = row
    if str(video_id) not in valid_video_ids:
        return f"Skipped {video_id} (not in dataset)"

    dir_prefix = f"{index:012d}"
    formatted_filename = f"{dir_prefix[:9]}/{dir_prefix}_{video_id}.mp4"
    file_path = os.path.join(base_path, formatted_filename)

    if os.path.exists(file_path):
        return f"Skipped {video_id} (already exists)"

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        res = requests.get(video_url, stream=True, timeout=60)
        res.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=10240):
                if chunk:
                    f.write(chunk)
        return f"Downloaded {video_id}"
    except Exception as e:
        return f"Failed {video_id}: {e}"

# === Step 5: Prepare data and run downloads ===
input_data = [(i, row["link"], str(row["videoId"])) for i, row in pexels_df.iterrows()]
max_workers = 4

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_video, row) for row in input_data]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
        result = f.result()
        # Optional: print(result)
