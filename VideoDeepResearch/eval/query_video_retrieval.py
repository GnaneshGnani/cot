import argparse
import json
import os
import sys

DEFAULT_HF_HOME = "/nfs-stor/ghazi.ahmad/HF_HOME"
DEFAULT_RETRIEVER_ASSETS_ROOT = "/home/ghazi.ahmad/model_assets"
DEFAULT_LANGUAGEBIND_VIDEO_MODEL_PATH = f"{DEFAULT_RETRIEVER_ASSETS_ROOT}/LanguageBind_Video_FT"
DEFAULT_LANGUAGEBIND_IMAGE_MODEL_PATH = f"{DEFAULT_RETRIEVER_ASSETS_ROOT}/LanguageBind_Image"
DEFAULT_LANGUAGEBIND_VIDEO_TOKENIZER_PATH = DEFAULT_LANGUAGEBIND_VIDEO_MODEL_PATH
DEFAULT_BGE_M3_MODEL_PATH = f"{DEFAULT_RETRIEVER_ASSETS_ROOT}/bge-m3"

os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("RETRIEVER_ASSETS_ROOT", DEFAULT_RETRIEVER_ASSETS_ROOT)
os.environ.setdefault("LANGUAGEBIND_VIDEO_MODEL_PATH", DEFAULT_LANGUAGEBIND_VIDEO_MODEL_PATH)
os.environ.setdefault("LANGUAGEBIND_IMAGE_MODEL_PATH", DEFAULT_LANGUAGEBIND_IMAGE_MODEL_PATH)
os.environ.setdefault("LANGUAGEBIND_VIDEO_TOKENIZER_PATH", DEFAULT_LANGUAGEBIND_VIDEO_TOKENIZER_PATH)
os.environ.setdefault("BGE_M3_MODEL_PATH", DEFAULT_BGE_M3_MODEL_PATH)

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import torch

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    VideoFileClip = None

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from retriever_languagebind import Retrieval_Manager


def get_video_duration(video_path: str) -> float:
    if VideoFileClip is None:
        raise RuntimeError("moviepy is not available")
    with VideoFileClip(video_path) as video:
        return float(video.duration)


def clip_path_to_segment(clip_path: str, score: float, duration: float, clip_duration: int):
    base = os.path.basename(clip_path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")

    if len(parts) >= 5 and parts[0] == "clip" and parts[3] == "to":
        start_h, start_m, start_s = [int(x) for x in parts[2].split("-")]
        end_h, end_m, end_s = [int(x) for x in parts[4].split("-")]
        start = float(start_h * 3600 + start_m * 60 + start_s)
        end = float(end_h * 3600 + end_m * 60 + end_s)
    else:
        clip_number = int(parts[1])
        start = float(clip_number * clip_duration)
        end = float(min(duration, start + clip_duration))

    return {
        "clip_path": clip_path,
        "start": start,
        "end": min(float(duration), end),
        "confidence": float(score),
    }

def main():
    parser = argparse.ArgumentParser(description="Run query-to-video-clip retrieval like the temporal-grounding pipeline.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--dataset-folder", default="./data")
    parser.add_argument("--clip-duration", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever-type", default="large")
    parser.add_argument("--clip-fps", type=float, default=2.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--languagebind-video-model-path", default="")
    parser.add_argument("--languagebind-image-model-path", default="")
    parser.add_argument("--languagebind-video-tokenizer-path", default="")
    parser.add_argument("--bge-m3-model-path", default="")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("RETRIEVER_ASSETS_ROOT", DEFAULT_RETRIEVER_ASSETS_ROOT)

    assets_root = os.environ["RETRIEVER_ASSETS_ROOT"]
    video_model_path = (
        args.languagebind_video_model_path
        or os.environ.get("LANGUAGEBIND_VIDEO_MODEL_PATH", "").strip()
        or f"{assets_root}/LanguageBind_Video_FT"
    )
    image_model_path = (
        args.languagebind_image_model_path
        or os.environ.get("LANGUAGEBIND_IMAGE_MODEL_PATH", "").strip()
        or f"{assets_root}/LanguageBind_Image"
    )
    tokenizer_path = (
        args.languagebind_video_tokenizer_path
        or os.environ.get("LANGUAGEBIND_VIDEO_TOKENIZER_PATH", "").strip()
        or video_model_path
    )
    bge_model_path = (
        args.bge_m3_model_path
        or os.environ.get("BGE_M3_MODEL_PATH", "").strip()
        or f"{assets_root}/bge-m3"
    )

    missing = []
    if not video_model_path:
        missing.append("LANGUAGEBIND_VIDEO_MODEL_PATH")
    if not image_model_path:
        missing.append("LANGUAGEBIND_IMAGE_MODEL_PATH")
    if not tokenizer_path:
        missing.append("LANGUAGEBIND_VIDEO_TOKENIZER_PATH")
    if not bge_model_path:
        missing.append("BGE_M3_MODEL_PATH")
    if missing:
        raise SystemExit(
            "Missing local model paths: "
            + ", ".join(missing)
            + f". Expected refiner defaults under {assets_root}."
        )

    os.environ["LANGUAGEBIND_VIDEO_MODEL_PATH"] = video_model_path
    os.environ["LANGUAGEBIND_IMAGE_MODEL_PATH"] = image_model_path
    os.environ["LANGUAGEBIND_VIDEO_TOKENIZER_PATH"] = tokenizer_path
    os.environ["BGE_M3_MODEL_PATH"] = bge_model_path

    class RetrieverArgs:
        dataset_folder = args.dataset_folder
        dataset = "demo"
        clip_duration = args.clip_duration
        retriever_type = args.retriever_type
        clip_fps = args.clip_fps

    duration = get_video_duration(args.video_path)
    clip_save_folder = f"{args.dataset_folder}/clips/{args.clip_duration}/"
    retriever = Retrieval_Manager(RetrieverArgs(), clip_save_folder=clip_save_folder)

    if torch.cuda.is_available() and args.gpu >= 0:
        retriever.load_model_to_gpu(args.gpu)
    else:
        retriever.load_model_to_cpu()

    folder_path = f"{args.dataset_folder}/embeddings/{args.clip_duration}/{args.retriever_type}"
    clip_paths, _ = retriever.calculate_video_clip_embedding(
        args.video_path,
        folder_path,
        total_duration=duration,
        pre_calculate=False,
    )
    if len(clip_paths) == 0:
        retriever.calculate_video_clip_embedding(
            args.video_path,
            folder_path,
            total_duration=duration,
            pre_calculate=True,
        )

    matches = retriever.get_informative_clips(
        args.query,
        video_path=args.video_path,
        top_k=args.top_k,
        total_duration=duration,
    )
    result = {
        "query": args.query,
        "video_path": args.video_path,
        "video_duration": duration,
        "retrieval_backend": "clip",
        "segments": [
            clip_path_to_segment(clip_path, score, duration, args.clip_duration)
            for clip_path, score in matches
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
