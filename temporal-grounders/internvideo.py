import cv2
import importlib.util
import os
from pathlib import Path

import math
import torch

HF_HOME = "/nfs-stor/ghazi.ahmad/HF_HOME"
HF_CACHE = f"{HF_HOME}/hub"
HF_MODULES_CACHE = f"{HF_HOME}/modules"

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
os.environ.setdefault("HF_MODULES_CACHE", HF_MODULES_CACHE)
Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
Path(HF_MODULES_CACHE).mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download
from transformers import AutoModel

REPO_ID = "OpenGVLab/InternVideo2-Stage2_6B"
HF_CACHE = Path(HF_HOME) / "hub"
video_path = "/nfs-stor/ghazi.ahmad/videos/5Jrv1h4AztM.mp4"
text_candidates = [
    "ice-packing process starts"
]

# Resolve the shared local snapshot directly so the script stays offline and
# uses the patched `modeling_internvideo2.py` under `/nfs-stor`.
snapshot_root = HF_CACHE / f"models--{REPO_ID.replace('/', '--')}" / "snapshots"
local_snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir()) if snapshot_root.exists() else []
repo_dir = str(local_snapshots[-1]) if local_snapshots else snapshot_download(
    repo_id=REPO_ID,
    cache_dir=str(HF_CACHE),
    local_files_only=True,
)

# Dynamically import helper functions from the downloaded model repo.
helper_path = Path(repo_dir) / "modeling_internvideo2.py"
spec = importlib.util.spec_from_file_location("modeling_internvideo2", helper_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

frames2tensor = mod.frames2tensor

# Load model from the downloaded repo snapshot.
model = AutoModel.from_pretrained(repo_dir, trust_remote_code=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
clip_duration = 10.0
top_k = 5


def sample_frames_uniform(video, start_frame: int, end_frame: int, num_frames: int):
    if end_frame < start_frame:
        end_frame = start_frame

    if num_frames <= 1:
        indices = [start_frame]
    else:
        indices = [
            round(start_frame + i * (end_frame - start_frame) / (num_frames - 1))
            for i in range(num_frames)
        ]

    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = video.read()
        if success:
            frames.append(frame)

    if not frames:
        raise RuntimeError("Failed to decode sampled frames.")

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def rank_video_clips(path: str, texts, clip_seconds: float, topk: int):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        fps = float(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Could not determine video metadata for: {path}")

        clip_count = max(1, math.ceil(total_frames / (fps * clip_seconds)))
        text_feats = torch.cat([model.get_txt_feat(text) for text in texts], dim=0)
        results = []

        for clip_idx in range(clip_count):
            start_sec = clip_idx * clip_seconds
            end_sec = min((clip_idx + 1) * clip_seconds, total_frames / fps)
            start_frame = min(total_frames - 1, max(0, round(start_sec * fps)))
            end_frame = min(total_frames - 1, max(start_frame, round(end_sec * fps) - 1))
            frames = sample_frames_uniform(video, start_frame, end_frame, int(model.config.num_frames))
            frames_tensor = frames2tensor(
                frames,
                fnum=int(model.config.num_frames),
                target_size=(int(model.config.size_t), int(model.config.size_t)),
                device=device,
            )
            vid_feat = model.get_vid_feat(frames_tensor)
            similarities = (vid_feat @ text_feats.T)[0]
            best_idx = int(similarities.argmax().item())
            results.append(
                {
                    "clip_index": clip_idx,
                    "start": start_sec,
                    "end": end_sec,
                    "text": texts[best_idx],
                    "similarity": float(similarities[best_idx].item()),
                }
            )

        results.sort(key=lambda item: item["similarity"], reverse=True)
        return results[:topk]
    finally:
        video.release()


for result in rank_video_clips(video_path, text_candidates, clip_duration, top_k):
    print(
        f"clip {result['clip_index']:03d} "
        f"[{result['start']:.2f}s - {result['end']:.2f}s] "
        f"similarity={result['similarity']:.4f} "
        f"text={result['text']}"
    )
