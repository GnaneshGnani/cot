#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import numpy as np
import torch
import whisperx
from faster_whisper import utils as faster_whisper_utils


@contextmanager
def _whisperx_torch_load_compat():
    original_torch_load = getattr(torch, "load", None)
    if original_torch_load is None:
        yield
        return

    def _compat_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _dedupe_paths(paths):
    deduped = []
    seen = set()
    for path in paths:
        raw = str(path or "").strip()
        if not raw:
            continue
        key = os.path.realpath(raw)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(raw)
    return deduped


def _sanitize_ld_library_path():
    current = os.environ.get("LD_LIBRARY_PATH", "")
    gpu_paths = []
    for name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            mod = importlib.import_module(name)
            gpu_paths.extend(list(getattr(mod, "__path__", []) or []))
        except Exception:
            continue

    keep = []
    for entry in current.split(":"):
        raw = entry.strip()
        if not raw:
            continue
        normalized = os.path.realpath(raw).lower()
        if "/site-packages/nvidia/" in normalized:
            keep.append(raw)
            continue
        if any(
            marker in normalized
            for marker in (
                "/miniconda",
                "/anaconda",
                "/miniforge",
                "/mambaforge",
                "/micromamba",
                "/conda/",
            )
        ):
            continue
        keep.append(raw)

    keep = _dedupe_paths(list(gpu_paths) + keep)
    if keep:
        os.environ["LD_LIBRARY_PATH"] = ":".join(keep)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _whisperx_download_root(cli_value: str | None = None):
    raw = str(cli_value or "").strip()
    if raw:
        return str(Path(raw).expanduser())
    for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        candidate = str(os.getenv(key, "") or "").strip()
        if candidate:
            return str(Path(candidate).expanduser())
    hf_home = str(os.getenv("HF_HOME", "") or "").strip()
    if hf_home:
        return str((Path(hf_home).expanduser() / "hub").resolve())
    return None


def _whisperx_hub_roots():
    roots = []
    for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        candidate = str(os.getenv(key, "") or "").strip()
        if candidate:
            roots.append(Path(candidate).expanduser())
    hf_home = str(os.getenv("HF_HOME", "") or "").strip()
    if hf_home:
        hf_root = Path(hf_home).expanduser()
        roots.append(hf_root / "hub")
        roots.append(hf_root)
    deduped = []
    seen = set()
    for root in roots:
        normalized = str(root.resolve()) if root.exists() else str(root)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(root)
    return deduped


def _whisperx_repo_id(model_name: str) -> str:
    raw_name = str(model_name or "").strip()
    if not raw_name:
        return "Systran/faster-whisper-small"
    if "/" in raw_name:
        return raw_name
    mapped = getattr(faster_whisper_utils, "_MODELS", {}).get(raw_name)
    return str(mapped or raw_name)


def _resolve_hf_snapshot(repo_id: str):
    raw_repo = str(repo_id or "").strip()
    if not raw_repo or "/" not in raw_repo:
        return None
    repo_dir_name = f"models--{raw_repo.replace('/', '--')}"
    for hub_root in _whisperx_hub_roots():
        repo_dir = hub_root / repo_dir_name
        snapshots_dir = repo_dir / "snapshots"
        ref_main = repo_dir / "refs" / "main"
        if ref_main.is_file():
            snapshot_id = ref_main.read_text(encoding="utf-8").strip()
            if snapshot_id:
                snapshot_path = snapshots_dir / snapshot_id
                if snapshot_path.is_dir():
                    return str(snapshot_path.resolve())
        if snapshots_dir.is_dir():
            snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
            if snapshot_dirs:
                return str(snapshot_dirs[-1].resolve())
    return None


def _whisperx_model_config(model_name: str, download_root: str | None = None, local_files_only: bool = False):
    requested = str(model_name or "").strip() or "small"
    direct_path = Path(requested).expanduser()
    if direct_path.exists():
        return {
            "requested_name": requested,
            "model_spec": str(direct_path.resolve()),
            "download_root": None,
            "local_files_only": True,
        }

    repo_id = _whisperx_repo_id(requested)
    cached_snapshot = _resolve_hf_snapshot(repo_id)
    if cached_snapshot:
        return {
            "requested_name": requested,
            "model_spec": cached_snapshot,
            "download_root": None,
            "local_files_only": True,
        }

    effective_local_only = bool(local_files_only)
    if effective_local_only:
        roots = [str(root) for root in _whisperx_hub_roots()] or ["<unset cache roots>"]
        raise RuntimeError(
            f"WhisperX model {requested!r} is not cached locally. "
            f"Expected a snapshot for {repo_id!r} under one of: {', '.join(roots)}. "
            "Set WHISPERX_MODEL to a local path, pre-download the model, "
            "or set WHISPERX_LOCAL_FILES_ONLY=0 to allow an on-demand download."
        )

    return {
        "requested_name": requested,
        "model_spec": repo_id,
        "download_root": _whisperx_download_root(download_root),
        "local_files_only": False,
    }


@contextmanager
def _whisperx_hub_access(allow_download: bool):
    if not allow_download:
        yield
        return
    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ffmpeg_binary_candidates():
    candidates = []
    for env_name in ("WHISPERX_FFMPEG_PATH", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        raw = str(os.getenv(env_name, "") or "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    which_hit = shutil.which("ffmpeg")
    if which_hit:
        candidates.append(Path(which_hit))
    try:
        import imageio_ffmpeg

        imageio_hit = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_hit:
            candidates.append(Path(imageio_hit))
    except Exception:
        pass
    conda_prefix = str(os.getenv("CONDA_PREFIX", "") or "").strip()
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "ffmpeg")
    candidates.append(Path(sys.executable).resolve().parent / "ffmpeg")
    candidates.extend(
        [
            Path("/apps/local/anaconda3/bin/ffmpeg"),
            Path("/usr/bin/ffmpeg"),
            Path("/bin/ffmpeg"),
        ]
    )

    deduped = []
    seen = set()
    for path in candidates:
        normalized = str(path.expanduser().resolve()) if path.expanduser().exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _resolve_ffmpeg_binary() -> str:
    for path in _ffmpeg_binary_candidates():
        candidate = Path(path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise FileNotFoundError(
        "ffmpeg binary not found. Set WHISPERX_FFMPEG_PATH or install ffmpeg."
    )


def _ffmpeg_ready_env(env=None):
    prepared = dict(env or os.environ)
    ffmpeg_bin = _resolve_ffmpeg_binary()
    ffmpeg_dir = str(Path(ffmpeg_bin).resolve().parent)
    current_path = str(prepared.get("PATH", "") or "")
    path_entries = [entry for entry in current_path.split(":") if entry]
    if ffmpeg_dir not in path_entries:
        prepared["PATH"] = f"{ffmpeg_dir}:{current_path}" if current_path else ffmpeg_dir
    prepared["FFMPEG_BINARY"] = ffmpeg_bin
    prepared["IMAGEIO_FFMPEG_EXE"] = ffmpeg_bin
    return prepared


def _load_audio_with_ffmpeg(video_path: str, sr: int = None) -> np.ndarray:
    sample_rate = int(sr or getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000))
    cmd = [
        _resolve_ffmpeg_binary(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        video_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            env=_ffmpeg_ready_env(),
        ).stdout
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode(errors="ignore").strip()
        raise RuntimeError(f"Failed to load audio with ffmpeg: {detail}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _transcribe_window(
    video_path: str,
    start_time: float,
    end_time: float,
    model_name: str,
    device: str,
    aux_device: str,
    compute_type: str,
    batch_size: int,
    language: str | None,
    download_root: str | None,
    local_files_only: bool,
):
    _sanitize_ld_library_path()
    transcribe_device_obj = torch.device(device)
    transcribe_device = transcribe_device_obj.type
    transcribe_device_index = int(transcribe_device_obj.index or 0)
    aux_device = str(torch.device(aux_device))
    model_config = _whisperx_model_config(
        model_name=model_name,
        download_root=download_root,
        local_files_only=local_files_only,
    )

    with _whisperx_torch_load_compat():
        vad_model = whisperx.asr.load_vad_model(torch.device(aux_device))

    with _whisperx_torch_load_compat():
        with _whisperx_hub_access(allow_download=not bool(model_config.get("local_files_only"))):
            model = whisperx.load_model(
                str(model_config["model_spec"]),
                transcribe_device,
                device_index=transcribe_device_index,
                compute_type=compute_type,
                vad_model=vad_model,
                download_root=model_config.get("download_root"),
                local_files_only=bool(model_config.get("local_files_only")),
            )

    audio = _load_audio_with_ffmpeg(video_path)
    sample_rate = float(getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000))
    sample_start = max(0, int(float(start_time) * sample_rate))
    sample_end = min(len(audio), int(float(end_time) * sample_rate))
    audio_window = audio[sample_start:sample_end] if sample_end > sample_start else audio[0:0]

    if len(audio_window) == 0:
        return {
            "language_detected": "unknown",
            "transcript": "",
            "full_transcript": "",
            "segments": [],
            "words": [],
            "asr_backend": "whisperx_sidecar",
            "asr_runner": "sidecar",
            "asr_device": str(transcribe_device_obj),
            "asr_device_index": transcribe_device_index,
            "asr_aux_device": aux_device,
            "asr_compute_type": compute_type,
            "whisperx_model": model_name,
        }

    transcribe_kwargs = {"batch_size": int(batch_size)}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_window, **transcribe_kwargs)
    lang = result.get("language") or "en"
    align_warning = None

    try:
        with _whisperx_torch_load_compat():
            model_a, meta = whisperx.load_align_model(language_code=lang, device=aux_device)
        aligned = whisperx.align(
            result["segments"],
            model_a,
            meta,
            audio_window,
            aux_device,
            return_char_alignments=False,
        )
        segs = aligned.get("segments", [])
    except Exception as ex:
        align_warning = f"{type(ex).__name__}: {ex}"
        segs = [
            {
                "start": float(s.get("start", 0)),
                "end": float(s.get("end", 0)),
                "text": str(s.get("text", "")),
                "score": 0.75,
                "words": [],
            }
            for s in result.get("segments", [])
        ]

    offset = float(start_time)
    normalized = []
    for seg in segs:
        item = dict(seg)
        item["start"] = float(item.get("start", 0)) + offset
        item["end"] = float(item.get("end", 0)) + offset
        words = []
        for word in item.get("words") or []:
            word_item = dict(word)
            if "start" in word_item:
                word_item["start"] = float(word_item.get("start", 0)) + offset
            if "end" in word_item:
                word_item["end"] = float(word_item.get("end", 0)) + offset
            words.append(word_item)
        item["words"] = words
        normalized.append(item)

    segments = []
    words_flat = []
    for seg in normalized:
        segments.append(
            {
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": str(seg.get("text", "")).strip(),
                "speaker": seg.get("speaker"),
                "confidence": float(seg.get("score", 0.9) or 0.9),
            }
        )
        for word in seg.get("words") or []:
            words_flat.append(
                {
                    "word": str(word.get("word", "")),
                    "start": float(word.get("start", 0)),
                    "end": float(word.get("end", 0)),
                    "confidence": float(word.get("score", 0.0) or 0.0),
                }
            )

    transcript = " ".join(seg["text"] for seg in segments).strip()
    payload = {
        "language_detected": lang,
        "transcript": transcript,
        "full_transcript": transcript,
        "segments": segments,
        "words": words_flat,
        "asr_backend": "whisperx_sidecar",
        "asr_runner": "sidecar",
        "asr_device": str(transcribe_device_obj),
        "asr_device_index": transcribe_device_index,
        "asr_aux_device": aux_device,
        "asr_compute_type": compute_type,
        "whisperx_model": model_name,
    }
    if align_warning:
        payload["align_warning"] = align_warning
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--model-name", default=os.getenv("WHISPERX_MODEL", "small"))
    parser.add_argument("--device", default=os.getenv("WHISPERX_DEVICE", "cuda:0"))
    parser.add_argument("--aux-device", default=os.getenv("WHISPERX_AUX_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.getenv("WHISPERX_COMPUTE_TYPE", "float16"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("WHISPERX_BATCH", "8")))
    parser.add_argument("--language", default=os.getenv("WHISPERX_LANGUAGE", "").strip() or None)
    parser.add_argument("--download-root", default=os.getenv("WHISPERX_DOWNLOAD_ROOT", "").strip() or None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=_truthy_env(
            "WHISPERX_LOCAL_FILES_ONLY",
            default=_truthy_env("HF_HUB_OFFLINE", default=False),
        ),
    )
    args = parser.parse_args()

    try:
        payload = _transcribe_window(
            video_path=args.video_path,
            start_time=float(args.start_time),
            end_time=float(args.end_time),
            model_name=str(args.model_name),
            device=str(args.device),
            aux_device=str(args.aux_device),
            compute_type=str(args.compute_type),
            batch_size=int(args.batch_size),
            language=args.language,
            download_root=args.download_root,
            local_files_only=bool(args.local_files_only),
        )
    except Exception as ex:
        print(
            json.dumps(
                {
                    "error_type": type(ex).__name__,
                    "error": str(ex),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
