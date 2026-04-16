import ast
import base64
import gc
import io
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from contextlib import contextmanager, nullcontext
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import numpy as np
import torch
import torchvision.transforms as T
import cv2
try:
    import whisperx
except Exception:
    whisperx = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from faster_whisper import utils as faster_whisper_utils
except Exception:
    faster_whisper_utils = None
try:
    from laion_clap import CLAP_Module
except Exception:
    CLAP_Module = None

from openai import OpenAI
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

import pytesseract
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

import refiner_debug
from refiner_utils import openai_chat_completion_limit_kwargs, openai_chat_temperature_kwargs
from refine_prompt import (
    action_recognizer_prompt,
    chart_analyzer_prompt,
    counter_prompt,
    dense_captioner_prompt,
    math_solver_prompt,
    ocr_prompt,
    spatial_grunder_prompt,
)
from video_utils import robust_eval

_WHISPERX_MODEL = None
_WHISPERX_MODEL_KEY = None
_WHISPERX_ALIGN = None
_WHISPERX_META = None
_PADDLE_OCR = None
_CLAP_MODULE = None
_CLAP_RUNTIME = {}
_WHISPERX_SIDECAR_RUNTIME = {}


@contextmanager
def _whisperx_torch_load_compat():
    """Preserve pre-PyTorch-2.6 checkpoint loading for WhisperX internals.

    WhisperX/pyannote loads a packaged VAD checkpoint that contains non-tensor
    OmegaConf objects. Under PyTorch 2.6, `torch.load(..., weights_only=None)`
    now behaves like `weights_only=True`, which rejects that checkpoint. Scope
    the compatibility override to WhisperX model initialization only.
    """
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


class RefinerToolsMixin:
    def _chart_torch_device(self):
        raw = str(getattr(self, "chart_device", "cuda:1")).strip()
        if raw.isdigit():
            return torch.device(f"cuda:{int(raw)}")
        return torch.device(raw)

    def _list_dense_frame_paths(self, dataset_folder: str, video_path: str):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        dense_dir = os.path.join(dataset_folder, "dense_frames", video_name)
        if not os.path.isdir(dense_dir):
            return [], dense_dir
        files = [
            f
            for f in os.listdir(dense_dir)
            if f.startswith("frame_") and f.lower().endswith(".png")
        ]
        files.sort(key=lambda x: float(x.replace("frame_", "").replace(".png", "")))
        return [os.path.join(dense_dir, f) for f in files], dense_dir

    @staticmethod
    def _timestamp_from_dense_frame_path(path: str) -> float:
        base = os.path.basename(path)
        num = base.replace("frame_", "").replace(".png", "")
        return float(num)

    def _whisperx_torch_device(self):
        raw = os.getenv("WHISPERX_DEVICE", str(getattr(self, "asr_device", "cuda:0"))).strip()
        if not raw or raw.lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if raw.isdigit():
            raw = f"cuda:{int(raw)}"
        try:
            device = torch.device(raw)
        except Exception:
            fallback = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"  WhisperX device {raw!r} is invalid; falling back to {fallback}.")
            return fallback
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("  WhisperX requested CUDA, but CUDA is unavailable; falling back to cpu.")
                return torch.device("cpu")
            if device.index is not None and device.index >= torch.cuda.device_count():
                fallback = torch.device("cuda:0")
                print(f"  WhisperX device {raw!r} is out of range; falling back to {fallback}.")
                return fallback
        return device

    def _whisperx_compute_type(self, device: torch.device) -> str:
        raw = os.getenv("WHISPERX_COMPUTE_TYPE", str(getattr(self, "asr_compute_type", "") or "")).strip()
        if raw:
            return raw
        return "float16" if device.type == "cuda" else "int8"

    def _whisperx_aux_device(self) -> str:
        raw = os.getenv("WHISPERX_AUX_DEVICE", "cpu").strip()
        if not raw:
            return "cpu"
        if raw.isdigit():
            return f"cuda:{int(raw)}"
        return raw

    def _env_flag(self, name: str, default: bool = False) -> bool:
        raw = str(os.getenv(name, "") or "").strip().lower()
        if not raw:
            return bool(default)
        return raw in {"1", "true", "yes", "on"}

    def _whisperx_download_root(self):
        raw = str(os.getenv("WHISPERX_DOWNLOAD_ROOT", "") or "").strip()
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

    def _whisperx_hub_roots(self):
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

    def _whisperx_repo_id(self, model_name: str) -> str:
        raw_name = str(model_name or "").strip()
        if not raw_name:
            return "Systran/faster-whisper-small"
        if "/" in raw_name:
            return raw_name
        mapped = getattr(faster_whisper_utils, "_MODELS", {}).get(raw_name)
        return str(mapped or raw_name)

    def _resolve_hf_snapshot(self, repo_id: str):
        raw_repo = str(repo_id or "").strip()
        if not raw_repo:
            return None
        repo_dir_name = f"models--{raw_repo.replace('/', '--')}"
        for hub_root in self._whisperx_hub_roots():
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

    def _resolve_cached_model_snapshot(self, repo_id: str, required_any=None):
        snapshot = self._resolve_hf_snapshot(repo_id)
        if snapshot is None:
            return None
        snapshot_path = Path(snapshot)
        if required_any:
            for name in required_any:
                if (snapshot_path / str(name)).is_file():
                    return str(snapshot_path.resolve())
            return None
        return str(snapshot_path.resolve())

    def _whisperx_model_config(self):
        requested = str(os.getenv("WHISPERX_MODEL", "small") or "").strip() or "small"
        direct_path = Path(requested).expanduser()
        if direct_path.exists():
            return {
                "requested_name": requested,
                "model_spec": str(direct_path.resolve()),
                "download_root": None,
                "local_files_only": True,
            }

        repo_id = self._whisperx_repo_id(requested)
        cached_snapshot = self._resolve_hf_snapshot(repo_id)
        if cached_snapshot:
            return {
                "requested_name": requested,
                "model_spec": cached_snapshot,
                "download_root": None,
                "local_files_only": True,
            }

        local_files_only = self._env_flag(
            "WHISPERX_LOCAL_FILES_ONLY",
            default=self._env_flag("HF_HUB_OFFLINE", default=False),
        )
        if local_files_only:
            roots = [str(root) for root in self._whisperx_hub_roots()] or ["<unset cache roots>"]
            raise RuntimeError(
                f"WhisperX model {requested!r} is not cached locally. "
                f"Expected a snapshot for {repo_id!r} under one of: {', '.join(roots)}. "
                "Set WHISPERX_MODEL to a local path, pre-download the model, "
                "or set WHISPERX_LOCAL_FILES_ONLY=0 to allow an on-demand download."
            )

        return {
            "requested_name": requested,
            "model_spec": repo_id,
            "download_root": self._whisperx_download_root(),
            "local_files_only": False,
        }

    @contextmanager
    def _whisperx_hub_access(self, allow_download: bool):
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

    def _ffmpeg_binary_candidates(self):
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
        python_bin = Path(sys.executable).resolve().parent / "ffmpeg"
        candidates.append(python_bin)
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

    def _resolve_ffmpeg_binary(self) -> str:
        for path in self._ffmpeg_binary_candidates():
            candidate = Path(path).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        raise FileNotFoundError(
            "ffmpeg binary not found. Set WHISPERX_FFMPEG_PATH or install ffmpeg."
        )

    def _ffmpeg_ready_env(self, env=None):
        prepared = dict(env or os.environ)
        ffmpeg_bin = self._resolve_ffmpeg_binary()
        ffmpeg_dir = str(Path(ffmpeg_bin).resolve().parent)
        current_path = str(prepared.get("PATH", "") or "")
        path_entries = [entry for entry in current_path.split(":") if entry]
        if ffmpeg_dir not in path_entries:
            prepared["PATH"] = f"{ffmpeg_dir}:{current_path}" if current_path else ffmpeg_dir
        prepared["FFMPEG_BINARY"] = ffmpeg_bin
        prepared["IMAGEIO_FFMPEG_EXE"] = ffmpeg_bin
        return prepared

    def _load_audio_with_ffmpeg(self, file_path: str):
        sample_rate = int(float(getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000)))
        cmd = [
            self._resolve_ffmpeg_binary(),
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file_path,
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
                env=self._ffmpeg_ready_env(),
            ).stdout
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or b"").decode(errors="ignore").strip()
            raise RuntimeError(f"Failed to load audio with ffmpeg: {detail}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def _whisperx_should_retry_on_cpu(self, error) -> bool:
        text = str(error or "").lower()
        markers = (
            "cuda failed with error",
            "cuda driver version is insufficient",
            "requested cuda, but cuda is unavailable",
            "device cuda is invalid",
            "no cuda-capable device is detected",
            "could not load library libcudnn",
            "libcudnn",
            "cudnn_ops_infer",
            "cublas",
            "libcuda",
            "subprocess for 'conda run",
            "aborted",
        )
        return any(marker in text for marker in markers)

    def _whisperx_ctranslate2_device(self, device_obj: torch.device):
        if device_obj.type == "cuda":
            return "cuda", int(device_obj.index or 0)
        return device_obj.type, 0

    def _whisperx_min_window_seconds(self) -> float:
        raw = os.getenv("WHISPERX_MIN_WINDOW", "").strip()
        try:
            value = float(raw) if raw else 8.0
        except Exception:
            value = 8.0
        duration = max(1.0, float(getattr(self, "duration", 1.0) or 1.0))
        return max(1.0, min(value, duration))

    def _whisperx_effective_range(self, start_time=None, end_time=None):
        requested_start, requested_end = self._get_time_range(start_time, end_time)
        transcribe_start = float(requested_start)
        transcribe_end = float(requested_end)
        min_window = self._whisperx_min_window_seconds()
        if (transcribe_end - transcribe_start) < min_window:
            mid = (transcribe_start + transcribe_end) / 2.0
            half = min_window / 2.0
            duration = float(getattr(self, "duration", transcribe_end) or transcribe_end)
            transcribe_start = max(0.0, mid - half)
            transcribe_end = min(duration, mid + half)
            if (transcribe_end - transcribe_start) < min_window:
                if transcribe_start <= 0.0:
                    transcribe_end = min(duration, min_window)
                else:
                    transcribe_start = max(0.0, transcribe_end - min_window)
        return {
            "requested_start": float(requested_start),
            "requested_end": float(requested_end),
            "transcribe_start": float(transcribe_start),
            "transcribe_end": float(transcribe_end),
            "expanded": (
                abs(float(transcribe_start) - float(requested_start)) > 1e-6
                or abs(float(transcribe_end) - float(requested_end)) > 1e-6
            ),
        }

    def _whisperx_extract_json_payload(self, text):
        parser = getattr(self, "_extract_json_payload", None)
        if callable(parser):
            parsed = parser(text)
            if isinstance(parsed, (dict, list)):
                return parsed

    def _clap_torch_device(self):
        raw = os.getenv("CLAP_DEVICE", "").strip()
        if not raw or raw.lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if raw.isdigit():
            raw = f"cuda:{int(raw)}"
        try:
            device = torch.device(raw)
        except Exception:
            fallback = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"  CLAP device {raw!r} is invalid; falling back to {fallback}.")
            return fallback
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("  CLAP requested CUDA, but CUDA is unavailable; falling back to cpu.")
                return torch.device("cpu")
            if device.index is not None and device.index >= torch.cuda.device_count():
                fallback = torch.device("cuda:0")
                print(f"  CLAP device {raw!r} is out of range; falling back to {fallback}.")
                return fallback
        return device

    def _clap_checkpoint_path(self):
        candidates = []
        raw = os.getenv("CLAP_CKPT_PATH", "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())

        hf_roots = []
        for env_name in ("HF_HOME", "VDR_SHARED_HF_HOME"):
            raw_root = str(os.getenv(env_name, "")).strip()
            if raw_root:
                hf_roots.append(Path(raw_root).expanduser())
        for hf_root in hf_roots:
            candidates.append(hf_root / "assets" / "laion_clap" / "630k-audioset-best.pt")
            snapshots_dir = hf_root / "hub" / "models--lukewys--laion_clap" / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot in sorted(snapshots_dir.iterdir()):
                    candidates.append(snapshot / "630k-audioset-best.pt")

        torch_home = str(os.getenv("TORCH_HOME", "")).strip()
        if torch_home:
            candidates.append(Path(torch_home).expanduser() / "hub" / "checkpoints" / "630k-audioset-best.pt")

        retriever_assets = str(os.getenv("RETRIEVER_ASSETS_ROOT", "")).strip()
        if retriever_assets:
            assets_root = Path(retriever_assets).expanduser()
            candidates.append(assets_root / "laion_clap" / "630k-audioset-best.pt")
            candidates.append(assets_root / "audio" / "laion_clap" / "630k-audioset-best.pt")

        conda_prefix = str(os.getenv("CONDA_PREFIX", "")).strip()
        if conda_prefix:
            prefix_root = Path(conda_prefix).expanduser()
            candidates.append(prefix_root / "share" / "laion_clap" / "630k-audioset-best.pt")
            candidates.append(prefix_root / "checkpoints" / "630k-audioset-best.pt")

        try:
            package_dir = Path(sys.modules[CLAP_Module.__module__].__file__).resolve().parent
            candidates.append(package_dir / "630k-audioset-best.pt")
            candidates.append(package_dir / "checkpoints" / "630k-audioset-best.pt")
        except Exception:
            pass

        seen = set()
        for path in candidates:
            normalized = str(path.resolve()) if path.exists() else str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            if path.is_file():
                return path
        return None

    def _clap_text_model_snapshot(self):
        return self._resolve_cached_model_snapshot(
            "roberta-base",
            required_any=(
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ),
        )

    def _clap_text_model_error(self) -> str:
        roots = [str(root) for root in self._whisperx_hub_roots()] or ["<unset cache roots>"]
        return (
            "LAION-CLAP requires cached roberta-base weights for its text branch, "
            "but no local model weights were found. Expected a roberta-base snapshot with "
            "`model.safetensors` or `pytorch_model.bin` under one of: "
            + ", ".join(roots)
        )

    def _audio_grounder_allow_targeted_heuristic_fallback(self) -> bool:
        raw = os.getenv("REFINER_AUDIO_GROUNDER_ALLOW_TARGETED_HEURISTIC", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _merge_audio_grounder_events(self, events):
        merged = []
        ordered = sorted(
            events or [],
            key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
        )
        for event in ordered:
            label = str(event.get("event_label", "")).strip()
            start = float(event.get("start", 0.0))
            end = float(event.get("end", start))
            confidence = float(event.get("confidence", 0.0) or 0.0)
            if not merged:
                merged.append(
                    {
                        "event_label": label,
                        "start": start,
                        "end": end,
                        "confidence": confidence,
                    }
                )
                continue
            prev = merged[-1]
            if prev["event_label"] == label and start <= float(prev["end"]) + 1e-6:
                prev["end"] = max(float(prev["end"]), end)
                prev["confidence"] = max(float(prev["confidence"]), confidence)
                continue
            merged.append(
                {
                    "event_label": label,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                }
            )
        return merged

    def _normalize_audio_label(self, text: str) -> str:
        cleaned = re.sub(r"[\[\]\(\)\{\}_\-]+", " ", str(text or "").strip().lower())
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _audio_grounder_query_mode(self, query: str) -> str:
        q = self._normalize_audio_label(query)
        if not q:
            return "inventory"
        broad_markers = (
            "distinct sound",
            "different sound",
            "sound effect",
            "sound effects",
            "audio cue",
            "audio cues",
            "non speech",
            "non speech sound",
            "non speech sounds",
            "non speech audio",
            "how many sounds",
            "what sounds",
            "which sounds",
            "various sounds",
        )
        if any(marker in q for marker in broad_markers):
            return "inventory"
        return "targeted"

    def _load_audio_window_with_ffmpeg(self, start_time=None, end_time=None, sample_rate: int = 16000):
        start, end = self._get_time_range(start_time, end_time)
        duration = max(0.05, float(end - start))
        cmd = [
            self._resolve_ffmpeg_binary(),
            "-nostdin",
            "-threads",
            "0",
            "-ss",
            str(float(start)),
            "-i",
            self.video_path,
            "-t",
            str(duration),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            "-",
        ]
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                env=self._ffmpeg_ready_env(),
            ).stdout
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or b"").decode(errors="ignore").strip()
            raise RuntimeError(f"Failed to extract audio window with ffmpeg: {detail}") from e

        audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        return audio, int(sample_rate), float(start), float(end)

    def _write_mono_wav(self, wav_path: str, audio_data, sample_rate: int) -> None:
        if audio_data is None:
            audio = np.asarray([], dtype=np.float32)
        else:
            audio = np.asarray(audio_data, dtype=np.float32).flatten()
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())

    def _audio_grounder_match_label(self, label: str, query: str) -> bool:
        normalized_label = self._normalize_audio_label(label)
        normalized_query = self._normalize_audio_label(query)
        if not normalized_label or not normalized_query:
            return False
        if normalized_label in normalized_query or normalized_query in normalized_label:
            return True

        stop = {
            "a", "an", "and", "audio", "background", "cue", "cues", "different",
            "distinct", "during", "effect", "effects", "find", "for", "in",
            "is", "non", "of", "or", "sound", "sounds", "speech", "the", "to",
            "when", "with",
        }
        label_tokens = {tok for tok in normalized_label.split() if tok not in stop}
        query_tokens = {tok for tok in normalized_query.split() if tok not in stop}
        if label_tokens and query_tokens and (label_tokens & query_tokens):
            return True

        alias_groups = [
            {"music", "melody", "song", "instrumental", "violin", "piano", "guitar"},
            {"applause", "clap", "clapping", "cheer", "cheering", "crowd"},
            {"bell", "doorbell", "ring", "ringing", "chime", "alarm", "beep"},
            {"glass", "breaking", "break", "smash", "crash", "bang", "thud", "slam", "impact"},
            {"engine", "motor", "rev", "revving", "vehicle", "car"},
            {"dog", "bark", "barking", "woof", "duck", "quack", "bird", "chirp", "cat", "meow", "animal"},
        ]
        for group in alias_groups:
            if (label_tokens & group) and (query_tokens & group):
                return True
        return False

    def _audio_grounder_from_subtitle_tags(self, arguments: dict) -> dict:
        query = str(arguments.get("query", "")).strip()
        mode = self._audio_grounder_query_mode(query)
        subtitle_result = self._get_asr_result_from_subtitles(
            arguments.get("start_time"), arguments.get("end_time")
        )
        segments = subtitle_result.get("segments") or []
        if not segments:
            return {
                "query": query,
                "query_mode": mode,
                "events": [],
                "audio_summary": "No subtitle-derived non-speech tags found in the requested range.",
                "backend": "subtitle_tags",
                "audio_status": "no_subtitle_tags",
            }

        explicit_music = {"music", "background music", "instrumental music", "song"}
        events = []
        for seg in segments:
            text = str(seg.get("text", "") or "").strip()
            if not text:
                continue
            tags = []
            for pattern in (r"\[([^\]]+)\]", r"\(([^\)]+)\)", r"♪([^♪]+)♪"):
                tags.extend(re.findall(pattern, text))
            normalized_text = self._normalize_audio_label(text)
            if "♪" in text and not tags:
                tags.append("music")
            if not tags and normalized_text in explicit_music:
                tags.append(normalized_text)

            for raw_tag in tags:
                clean_tag = re.sub(r"\s+", " ", str(raw_tag or "").strip()).strip(" -:;,.")
                if not clean_tag:
                    continue
                if mode == "targeted" and not self._audio_grounder_match_label(clean_tag, query):
                    continue
                events.append(
                    {
                        "event_label": clean_tag,
                        "start": float(seg.get("start", 0.0) or 0.0),
                        "end": float(seg.get("end", 0.0) or 0.0),
                        "confidence": 1.0,
                    }
                )

        merged = self._merge_audio_grounder_events(events)
        distinct_groups = []
        by_label = {}
        for event in merged:
            key = event["event_label"]
            bucket = by_label.setdefault(key, [])
            bucket.append(event)
        for label, members in sorted(by_label.items(), key=lambda item: min(x["start"] for x in item[1])):
            distinct_groups.append(
                {
                    "event_label": label,
                    "count": len(members),
                    "start": min(float(x["start"]) for x in members),
                    "end": max(float(x["end"]) for x in members),
                    "confidence": max(float(x.get("confidence", 0.0) or 0.0) for x in members),
                }
            )
        summary = (
            f"Subtitle tags indicate {len(distinct_groups)} distinct non-speech sound types."
            if distinct_groups
            else "Subtitles are available, but no matching non-speech sound tags were found for this query."
        )
        return {
            "query": query,
            "query_mode": mode,
            "events": merged,
            "distinct_event_groups": distinct_groups,
            "audio_summary": summary,
            "backend": "subtitle_tags",
            "audio_status": "ok" if merged else "no_match",
        }

    def _audio_grounder_match_heuristic_label(self, event_label: str, query: str) -> bool:
        label = self._normalize_audio_label(event_label)
        q = self._normalize_audio_label(query)
        if not label or not q:
            return False
        if self._audio_grounder_match_label(label, q):
            return True

        query_to_labels = {
            "music": {"sustained tonal sound", "tonal chime or ring"},
            "melody": {"sustained tonal sound", "tonal chime or ring"},
            "song": {"sustained tonal sound"},
            "violin": {"sustained tonal sound"},
            "piano": {"sustained tonal sound"},
            "bell": {"tonal chime or ring"},
            "ring": {"tonal chime or ring"},
            "chime": {"tonal chime or ring"},
            "alarm": {"tonal chime or ring", "broadband noise"},
            "beep": {"tonal chime or ring"},
            "engine": {"low hum or engine"},
            "motor": {"low hum or engine"},
            "rev": {"low hum or engine"},
            "glass": {"sharp broadband impact", "percussive noise burst"},
            "break": {"sharp broadband impact", "percussive noise burst"},
            "crash": {"sharp broadband impact", "percussive noise burst"},
            "bang": {"sharp broadband impact", "percussive noise burst"},
            "slam": {"sharp broadband impact", "percussive noise burst"},
            "impact": {"sharp broadband impact", "percussive noise burst"},
            "dog": {"animal like call"},
            "bark": {"animal like call"},
            "duck": {"animal like call"},
            "quack": {"animal like call"},
            "bird": {"animal like call"},
            "chirp": {"animal like call"},
            "cat": {"animal like call"},
            "meow": {"animal like call"},
            "animal": {"animal like call"},
            "applause": {"percussive noise burst", "broadband noise"},
            "clap": {"percussive noise burst"},
            "cheer": {"broadband noise", "percussive noise burst"},
            "crowd": {"broadband noise", "percussive noise burst"},
            "whoosh": {"whoosh or swish"},
            "swish": {"whoosh or swish"},
        }
        for token, labels in query_to_labels.items():
            if token in q and label in labels:
                return True
        return False

    def _audio_grounder_heuristic(self, arguments: dict) -> dict:
        query = str(arguments.get("query", "")).strip()
        mode = self._audio_grounder_query_mode(query)
        audio_data, sr, start, end = self._load_audio_window_with_ffmpeg(
            arguments.get("start_time"),
            arguments.get("end_time"),
            sample_rate=16000,
        )
        if len(audio_data) == 0:
            return {
                "query": query,
                "query_mode": mode,
                "events": [],
                "audio_summary": "The requested audio window is empty after extraction.",
                "backend": "heuristic_audio",
                "audio_status": "empty_window",
            }

        frame_sec = 0.064
        hop_sec = 0.032
        frame_len = max(256, int(frame_sec * sr))
        hop_len = max(128, int(hop_sec * sr))
        if len(audio_data) < frame_len:
            padded = np.zeros(frame_len, dtype=np.float32)
            padded[: len(audio_data)] = audio_data
            audio_data = padded

        window = np.hanning(frame_len).astype(np.float32)
        rms_values = []
        descriptors = []
        cursor = 0
        while cursor + frame_len <= len(audio_data):
            frame = audio_data[cursor : cursor + frame_len]
            weighted = frame * window
            spectrum = np.abs(np.fft.rfft(weighted)) + 1e-8
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            total = float(spectrum.sum())
            centroid = float((freqs * spectrum).sum() / total)
            bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spectrum).sum() / total))
            flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
            rms = float(np.sqrt(np.mean(frame ** 2) + 1e-10))
            signs = np.sign(frame)
            zero_cross = float(np.mean(np.abs(np.diff(signs)) > 0))
            rms_values.append(rms)
            descriptors.append(
                {
                    "time": cursor / float(sr),
                    "rms": rms,
                    "centroid": centroid,
                    "bandwidth": bandwidth,
                    "flatness": flatness,
                    "zero_cross": zero_cross,
                }
            )
            cursor += hop_len

        if not descriptors:
            return {
                "query": query,
                "query_mode": mode,
                "events": [],
                "audio_summary": "The requested audio window is too short for heuristic analysis.",
                "backend": "heuristic_audio",
                "audio_status": "too_short",
            }

        rms_arr = np.asarray(rms_values, dtype=np.float32)
        smooth_rms = np.convolve(rms_arr, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        baseline = float(np.percentile(smooth_rms, 25))
        high = float(np.percentile(smooth_rms, 85))
        threshold = max(0.008, baseline + 0.35 * max(0.0, high - baseline))
        active = smooth_rms >= threshold

        gap_frames = max(1, int(round(0.10 / hop_sec)))
        min_frames = max(1, int(round(0.08 / hop_sec)))
        for idx in range(1, len(active) - 1):
            if not active[idx] and active[max(0, idx - gap_frames) : idx].any() and active[idx + 1 : min(len(active), idx + gap_frames + 1)].any():
                active[idx] = True

        raw_events = []
        idx = 0
        while idx < len(active):
            if not active[idx]:
                idx += 1
                continue
            start_idx = idx
            while idx < len(active) and active[idx]:
                idx += 1
            end_idx = idx
            if (end_idx - start_idx) < min_frames:
                continue
            block = descriptors[start_idx:end_idx]
            if not block:
                continue
            event_start = float(start + block[0]["time"])
            event_end = float(start + block[-1]["time"] + frame_sec)
            duration = max(0.0, event_end - event_start)
            mean_centroid = float(np.mean([x["centroid"] for x in block]))
            mean_bandwidth = float(np.mean([x["bandwidth"] for x in block]))
            mean_flatness = float(np.mean([x["flatness"] for x in block]))
            mean_zcr = float(np.mean([x["zero_cross"] for x in block]))
            peak_rms = float(np.max([x["rms"] for x in block]))

            if duration >= 0.65 and mean_flatness < 0.20 and mean_bandwidth < 1200 and mean_centroid < 900:
                label = "low hum or engine"
            elif duration >= 0.25 and mean_flatness < 0.22 and mean_bandwidth < 1500 and mean_centroid > 1200:
                label = "tonal chime or ring"
            elif 0.12 <= duration <= 0.7 and mean_flatness < 0.32 and 500 <= mean_centroid <= 2600 and mean_bandwidth < 1800:
                label = "animal like call"
            elif duration < 0.35 and mean_flatness > 0.42 and mean_centroid > 1800:
                label = "sharp broadband impact"
            elif duration < 0.6 and mean_bandwidth > 2200 and mean_flatness > 0.22:
                label = "whoosh or swish"
            elif duration < 0.5 and mean_flatness > 0.30:
                label = "percussive noise burst"
            elif duration >= 0.45 and mean_flatness < 0.28:
                label = "sustained tonal sound"
            elif duration >= 0.45:
                label = "broadband noise"
            else:
                label = "generic non speech sound"

            confidence = min(0.95, max(0.25, peak_rms / (float(np.max(smooth_rms)) + 1e-8)))
            raw_events.append(
                {
                    "event_label": label,
                    "start": event_start,
                    "end": min(float(end), event_end),
                    "confidence": float(confidence),
                }
            )

        if mode == "targeted":
            filtered_events = [
                event for event in raw_events
                if self._audio_grounder_match_heuristic_label(event["event_label"], query)
            ]
        else:
            filtered_events = raw_events

        merged = self._merge_audio_grounder_events(filtered_events)
        distinct_groups = []
        by_label = {}
        for event in merged:
            key = event["event_label"]
            bucket = by_label.setdefault(key, [])
            bucket.append(event)
        for label, members in sorted(by_label.items(), key=lambda item: min(x["start"] for x in item[1])):
            distinct_groups.append(
                {
                    "event_label": label,
                    "count": len(members),
                    "start": min(float(x["start"]) for x in members),
                    "end": max(float(x["end"]) for x in members),
                    "confidence": max(float(x.get("confidence", 0.0) or 0.0) for x in members),
                }
            )

        if mode == "inventory":
            summary = (
                f"Heuristic audio analysis found {len(distinct_groups)} distinct non-speech sound types "
                f"in the requested interval."
            )
        elif merged:
            summary = f"Heuristic audio analysis found {len(merged)} matching non-speech events for the query."
        else:
            summary = "Heuristic audio analysis found non-speech activity, but none matched the requested sound query."

        return {
            "query": query,
            "query_mode": mode,
            "events": merged,
            "distinct_event_groups": distinct_groups,
            "raw_event_count": len(raw_events),
            "audio_summary": summary,
            "backend": "heuristic_audio",
            "audio_status": (
                "ok"
                if (merged or distinct_groups)
                else ("analyzed_no_match" if raw_events else "no_match")
            ),
        }

    def _whisperx_dedupe_paths(self, paths):
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

    def _whisperx_path_looks_like_conda_runtime(self, path: str) -> bool:
        normalized = os.path.realpath(path).lower()
        markers = (
            "/miniconda",
            "/anaconda",
            "/miniforge",
            "/mambaforge",
            "/micromamba",
            "/conda/",
        )
        return any(marker in normalized for marker in markers)

    def _whisperx_sidecar_runtime(self, base_cmd):
        cache_key = tuple(base_cmd)
        if cache_key not in _WHISPERX_SIDECAR_RUNTIME:
            probe = subprocess.run(
                base_cmd
                + [
                    "-c",
                    (
                        "import importlib, json, sys\n"
                        "paths=[]\n"
                        "for name in ('nvidia.cublas.lib', 'nvidia.cudnn.lib'):\n"
                        "    try:\n"
                        "        mod=importlib.import_module(name)\n"
                        "        paths.extend(list(getattr(mod, '__path__', []) or []))\n"
                        "    except Exception:\n"
                        "        pass\n"
                        "print(json.dumps({'sys_prefix': sys.prefix, 'gpu_ld_paths': paths}))\n"
                    ),
                ],
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
                check=False,
            )
            payload = self._whisperx_extract_json_payload((probe.stdout or "").strip())
            if not isinstance(payload, dict):
                payload = {}
            _WHISPERX_SIDECAR_RUNTIME[cache_key] = {
                "sys_prefix": str(payload.get("sys_prefix") or "").strip(),
                "gpu_ld_paths": self._whisperx_dedupe_paths(payload.get("gpu_ld_paths") or []),
            }
        return _WHISPERX_SIDECAR_RUNTIME[cache_key]

    def _whisperx_sidecar_base_cmd(self):
        python_bin = os.getenv("WHISPERX_PYTHON", "").strip()
        if python_bin:
            return [python_bin]

        env_name = os.getenv("WHISPERX_CONDA_ENV", "").strip()
        if not env_name:
            return None

        conda_exe = (
            os.getenv("CONDA_EXE", "").strip()
            or shutil.which("conda")
            or "/home/ghazi/miniconda3/bin/conda"
        )
        return [conda_exe, "run", "--no-capture-output", "-n", env_name, "python"]

    def _whisperx_sidecar_env(self, base_cmd):
        runtime = self._whisperx_sidecar_runtime(base_cmd)
        gpu_ld_paths = list(runtime.get("gpu_ld_paths") or [])
        env = self._ffmpeg_ready_env()
        filtered_current_ld = []
        for entry in str(env.get("LD_LIBRARY_PATH", "") or "").split(":"):
            raw = entry.strip()
            if not raw:
                continue
            real = os.path.realpath(raw)
            if any(real == os.path.realpath(path) for path in gpu_ld_paths):
                continue
            if self._whisperx_path_looks_like_conda_runtime(real):
                continue
            filtered_current_ld.append(raw)
        final_ld = self._whisperx_dedupe_paths(gpu_ld_paths + filtered_current_ld)
        if final_ld:
            env["LD_LIBRARY_PATH"] = ":".join(final_ld)
        else:
            env.pop("LD_LIBRARY_PATH", None)
        return env

    def _get_asr_whisperx_sidecar(self, start_time=None, end_time=None):
        base_cmd = self._whisperx_sidecar_base_cmd()
        if not base_cmd:
            raise RuntimeError("WhisperX sidecar is not configured")

        window = self._whisperx_effective_range(start_time, end_time)
        model_config = self._whisperx_model_config()
        sidecar_script = os.path.join(_eval_dir, "whisperx_sidecar.py")
        language_hint = os.getenv("WHISPERX_LANGUAGE", "").strip()
        batch_size = str(int(os.getenv("WHISPERX_BATCH", "8")))
        sidecar_env = self._whisperx_sidecar_env(base_cmd)

        def build_cmd(active_device: str, active_compute_type: str):
            cmd = base_cmd + [
                sidecar_script,
                "--video-path",
                self.video_path,
                "--start-time",
                str(window["transcribe_start"]),
                "--end-time",
                str(window["transcribe_end"]),
                "--model-name",
                str(model_config["model_spec"]),
                "--device",
                active_device,
                "--aux-device",
                self._whisperx_aux_device(),
                "--compute-type",
                active_compute_type,
                "--batch-size",
                batch_size,
            ]
            if model_config.get("download_root"):
                cmd.extend(["--download-root", str(model_config["download_root"])])
            if bool(model_config.get("local_files_only")):
                cmd.append("--local-files-only")
            if language_hint:
                cmd.extend(["--language", language_hint])
            return cmd

        def run_cmd(cmd):
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=sidecar_env,
                check=False,
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode != 0:
                err_payload = self._whisperx_extract_json_payload(stderr)
                if isinstance(err_payload, dict) and err_payload.get("error"):
                    detail = f'{err_payload.get("error_type", "RuntimeError")}: {err_payload.get("error")}'
                else:
                    detail = stderr or stdout or f"sidecar exited with code {proc.returncode}"
                raise RuntimeError(detail)
            return stdout

        device_obj = self._whisperx_torch_device()
        compute_type = self._whisperx_compute_type(device_obj)
        try:
            stdout = run_cmd(build_cmd(str(device_obj), compute_type))
        except RuntimeError as exc:
            if device_obj.type != "cuda":
                raise
            original_error = exc
            should_retry = self._whisperx_should_retry_on_cpu(exc)
            if should_retry:
                print(f"  WhisperX CUDA failed ({exc}); retrying ASR on cpu.")
            else:
                print(f"  WhisperX initial CUDA run failed ({exc}); attempting cpu fallback.")
            try:
                stdout = run_cmd(build_cmd("cpu", "int8"))
            except RuntimeError:
                raise original_error

        if not stdout:
            raise RuntimeError("WhisperX sidecar returned no stdout")
        result = self._whisperx_extract_json_payload(stdout)
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid JSON from WhisperX sidecar stdout: {stdout[:500]}")
        result.setdefault(
            "requested_range",
            {
                "start": float(window["requested_start"]),
                "end": float(window["requested_end"]),
            },
        )
        result.setdefault(
            "transcription_range",
            {
                "start": float(window["transcribe_start"]),
                "end": float(window["transcribe_end"]),
            },
        )
        result["window_expanded"] = bool(window["expanded"])
        return result

    def _chart_effective_api_bases(self):
        b = getattr(self, "chart_api_base", None)
        if b is not None and len(b) and (b[0] or "").strip():
            return list(b)
        return list(getattr(self, "planner_api_base", None) or [])

    def _chart_effective_api_keys(self):
        k = getattr(self, "chart_api_keys", None)
        if k is not None:
            return list(k)
        return list(getattr(self, "planner_api_keys", None) or [])

    def _math_solver_effective_model_name(self):
        model_name = str(os.getenv("MATH_SOLVER_MODEL_NAME", "") or "").strip()
        if model_name:
            return model_name
        vlm_model_name = str(getattr(self, "vlm_model_name", "") or "").strip()
        if vlm_model_name:
            return vlm_model_name
        local_vlm_model_name = str(getattr(self, "local_vlm_model_name", "") or "").strip()
        if local_vlm_model_name:
            return local_vlm_model_name
        return "Qwen/Qwen3-VL-8B-Instruct"

    def _math_solver_effective_api_bases(self):
        raw = str(os.getenv("MATH_SOLVER_API_BASE", "") or "").strip()
        if raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
        return list(getattr(self, "planner_api_base", None) or [])

    def _math_solver_effective_api_keys(self):
        raw = str(os.getenv("MATH_SOLVER_API_KEY", "") or "").strip()
        if raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
        return list(getattr(self, "planner_api_keys", None) or [])

    def _math_solver_has_explicit_backend(self) -> bool:
        return bool(str(os.getenv("MATH_SOLVER_API_BASE", "") or "").strip())

    def _math_solver_use_http(self, api_bases: list | None = None) -> bool:
        return self._math_solver_has_explicit_backend()

    def _call_math_solver_text(self, prompt: str) -> str:
        model_name = self._math_solver_effective_model_name()
        api_bases = self._math_solver_effective_api_bases()
        api_keys = self._math_solver_effective_api_keys()

        if self._math_solver_use_http(api_bases):
            normalized_messages = [{"role": "user", "content": str(prompt or "")}]
            pairs = list(zip(api_bases, api_keys))
            if not pairs:
                return ""

            for base, key in pairs:
                try:
                    client = OpenAI(base_url=base.strip(), api_key=key.strip())
                    request_kwargs = {
                        "model": model_name,
                        "messages": normalized_messages,
                        **openai_chat_temperature_kwargs(model_name, 0.0),
                        **openai_chat_completion_limit_kwargs(model_name, 2048),
                    }
                    try:
                        completion = client.chat.completions.create(
                            response_format={"type": "json_object"},
                            **request_kwargs,
                        )
                    except Exception:
                        completion = client.chat.completions.create(**request_kwargs)
                    out = completion.choices[0].message.content
                    return out if isinstance(out, str) else (out or "")
                except Exception as e:
                    print(f"[Math Solver] text call failed base={base} model={model_name}: {e}")
            return ""

        summarize = getattr(self, "_vlm_summarize_text", None)
        if callable(summarize):
            return summarize(prompt)
        return ""

    def _call_math_solver_with_frames(self, prompt: str, frame_paths: list, frame_timestamps: list) -> str:
        frame_paths = [str(path).strip() for path in (frame_paths or []) if str(path).strip()]
        if not frame_paths:
            return ""

        normalized_timestamps = []
        for idx, path in enumerate(frame_paths):
            ts = frame_timestamps[idx] if idx < len(frame_timestamps) else None
            ts = self._safe_float(ts, None)
            if ts is None:
                try:
                    ts = float(self._timestamp_from_dense_frame_path(path))
                except Exception:
                    ts = float(idx)
            normalized_timestamps.append(float(ts))

        model_name = self._math_solver_effective_model_name()
        api_bases = self._math_solver_effective_api_bases()
        api_keys = self._math_solver_effective_api_keys()

        if self._math_solver_use_http(api_bases):
            content = []
            for frame_path in frame_paths:
                if not os.path.exists(frame_path):
                    continue
                try:
                    image = Image.open(frame_path)
                    image.verify()
                    image = Image.open(frame_path)
                    width, height = image.size
                    if max(width, height) > 768:
                        if width > height:
                            new_width = 768
                            new_height = int(height * (768 / width))
                        else:
                            new_height = 768
                            new_width = int(width * (768 / height))
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    image.convert("RGB").save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    )
                except Exception as e:
                    print(f"[Math Solver] skipping invalid frame {frame_path}: {e}")

            if content:
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                for base, key in zip(api_bases, api_keys):
                    try:
                        client = OpenAI(base_url=base.strip(), api_key=key.strip())
                        request_kwargs = {
                            "model": model_name,
                            "messages": messages,
                            **openai_chat_temperature_kwargs(model_name, 0.0),
                            **openai_chat_completion_limit_kwargs(model_name, 2048),
                        }
                        try:
                            completion = client.chat.completions.create(
                                response_format={"type": "json_object"},
                                **request_kwargs,
                            )
                        except Exception:
                            completion = client.chat.completions.create(**request_kwargs)
                        out = completion.choices[0].message.content
                        return out if isinstance(out, str) else (out or "")
                    except Exception as e:
                        print(f"[Math Solver] vision call failed base={base} model={model_name}: {e}")

        outputs = self._batch_video2text([(prompt, frame_paths, normalized_timestamps)], force_local=False)
        if not outputs:
            return ""
        out = outputs[0]
        return out if isinstance(out, str) else str(out or "")

    def _score_chart_analysis_result(self, result: dict) -> float:
        if not isinstance(result, dict):
            return -1.0
        score = 0.0
        query_response = str(result.get("query_response", "") or "").strip().lower()
        if query_response.startswith("chart_analyzer error:") or query_response.startswith(
            "chart_analyzer unavailable"
        ):
            score -= 10.0
        if str(result.get("chart_type", "") or "").strip() not in {"", "unknown", "other"}:
            score += 3.0
        if str(result.get("title", "") or "").strip():
            score += 1.0
        score += min(float(len(result.get("series") or [])) * 3.0, 9.0)
        score += min(float(len(result.get("key_observations") or [])) * 0.75, 3.0)
        score += min(float(len(result.get("relationships") or [])) * 0.5, 2.0)
        score -= max(float(len(result.get("key_observations") or [])) - 6.0, 0.0) * 0.25
        score -= max(float(len(result.get("relationships") or [])) - 4.0, 0.0) * 0.5
        if query_response:
            score += 1.0
            if query_response.endswith((" and", " or", " to", " at", " of", ",", ":", ";")):
                score -= 1.0
        return score

    def _normalize_chart_text(self, value) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _chart_relationship_key(self, rel) -> tuple[str, str, str]:
        if not isinstance(rel, dict):
            return ("", "", "")
        return (
            self._normalize_chart_text(rel.get("from")),
            self._normalize_chart_text(rel.get("to")),
            self._normalize_chart_text(rel.get("label")),
        )

    def _is_diagram_label_claim(self, text: str) -> bool:
        normalized = self._normalize_chart_text(text)
        if not normalized:
            return False
        return any(
            token in normalized
            for token in (
                " label ",
                " labeled ",
                " marked ",
                " marking ",
                " text label ",
                " numeric label ",
                " is labeled",
                " are labeled",
                " labeled with",
                " marked with",
            )
        ) or normalized.startswith(("label ", "labels ", "labeled "))

    def _quoted_or_numeric_label_tokens(self, text: str) -> list[str]:
        tokens = []
        for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", str(text or "")):
            token = (left or right or "").strip()
            if token:
                tokens.append(token)
        if tokens:
            return tokens
        return re.findall(r"\b\d+(?:\.\d+)?\b", str(text or ""))

    def _collect_text_fields(self, value) -> list[str]:
        out = []
        if isinstance(value, dict):
            text = value.get("text")
            if text is not None:
                text = str(text).strip()
                if text:
                    out.append(text)
            for child in value.values():
                out.extend(self._collect_text_fields(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                out.extend(self._collect_text_fields(child))
        return out

    def _sanitize_math_solver_evidence_items(self, evidence_items: list[str]) -> list[str]:
        cleaned_items = [str(item or "").strip() for item in (evidence_items or []) if str(item or "").strip()]
        support_chunks = []
        for item in cleaned_items:
            raw_item = str(item).strip()
            parsed = None
            if raw_item.startswith(("[", "{")):
                try:
                    parsed = json.loads(raw_item)
                except Exception:
                    try:
                        parsed = ast.literal_eval(raw_item)
                    except Exception:
                        parsed = None
            if parsed is None:
                support_chunks.append(raw_item)
            else:
                support_chunks.extend(self._collect_text_fields(parsed))
        ocr_support_text = self._normalize_chart_text("\n".join(support_chunks))
        sanitized = []
        dropped_label_claims = 0

        for item in cleaned_items:
            raw_item = str(item).strip()
            parsed = None
            if raw_item.startswith("["):
                try:
                    parsed = json.loads(raw_item)
                except Exception:
                    try:
                        parsed = ast.literal_eval(raw_item)
                    except Exception:
                        parsed = None

            if isinstance(parsed, (list, tuple)):
                rewritten = []
                for entry in parsed:
                    text = str(entry or "").strip()
                    if not text:
                        continue
                    if self._is_diagram_label_claim(text):
                        tokens = [
                            self._normalize_chart_text(token)
                            for token in self._quoted_or_numeric_label_tokens(text)
                            if self._normalize_chart_text(token)
                        ]
                        if not tokens or not all(token in ocr_support_text for token in tokens):
                            dropped_label_claims += 1
                            continue
                    rewritten.append(text)
                if rewritten:
                    sanitized.append(json.dumps(rewritten, ensure_ascii=False))
                continue

            if self._is_diagram_label_claim(raw_item):
                tokens = [
                    self._normalize_chart_text(token)
                    for token in self._quoted_or_numeric_label_tokens(raw_item)
                    if self._normalize_chart_text(token)
                ]
                if not tokens or not all(token in ocr_support_text for token in tokens):
                    dropped_label_claims += 1
                    continue
            sanitized.append(raw_item)

        if dropped_label_claims > 0:
            sanitized.append(
                "Some diagram label-attachment or repeated-label claims from structured diagram reading "
                "were omitted because OCR did not corroborate the exact visible label text."
            )
        return sanitized

    def _strip_math_solver_answer_choice_text(self, text: str, choices: list[str] | None = None) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        normalized_choices = [str(choice or "").strip() for choice in (choices or []) if str(choice or "").strip()]
        parsed = None
        if cleaned.startswith(("[", "{")):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                try:
                    parsed = ast.literal_eval(cleaned)
                except Exception:
                    parsed = None

        if isinstance(parsed, (list, tuple, dict)):
            def _scrub_structured(value):
                if isinstance(value, str):
                    return self._strip_math_solver_answer_choice_text(value, normalized_choices)
                if isinstance(value, dict):
                    rewritten = {}
                    for key, child in value.items():
                        scrubbed_child = _scrub_structured(child)
                        if scrubbed_child in ("", None, [], {}):
                            continue
                        rewritten[key] = scrubbed_child
                    return rewritten
                if isinstance(value, (list, tuple)):
                    rewritten = []
                    for child in value:
                        scrubbed_child = _scrub_structured(child)
                        if scrubbed_child in ("", None, [], {}):
                            continue
                        rewritten.append(scrubbed_child)
                    return rewritten
                return value

            scrubbed = _scrub_structured(parsed)
            if scrubbed in ("", None, [], {}):
                return ""
            return json.dumps(scrubbed, ensure_ascii=False)

        marker_match = re.search(r"(?is)\b(?:answer\s+choices?|choices?|options?)\s*:", cleaned)
        if marker_match:
            cleaned = cleaned[:marker_match.start()].rstrip(" \t\r\n,;:-")

        instruction_match = re.search(
            r"(?is)\b(?:choose|select|pick|map)\b[^.\n]{0,140}\b(?:option|answer choice)s?\b.*$",
            cleaned,
        )
        if instruction_match:
            cleaned = cleaned[:instruction_match.start()].rstrip(" \t\r\n,;:-")

        lowered = cleaned.lower()
        choice_hits = []
        for choice in normalized_choices:
            position = lowered.find(choice.lower())
            if position >= 0:
                choice_hits.append(position)
        if len(choice_hits) >= 2:
            cleaned = cleaned[: min(choice_hits)].rstrip(" \t\r\n,;:-")
        elif choice_hits and re.search(r"(?is)\b(?:option|answer choice)s?\b", cleaned):
            cleaned = cleaned[: min(choice_hits)].rstrip(" \t\r\n,;:-")

        return cleaned.strip()

    def _split_answer_choice_label(self, choice: str) -> tuple[str | None, str]:
        text = str(choice or "").strip()
        if not text:
            return None, ""

        patterns = [
            r"^\(?\s*([A-Z])\s*\)?[\.\):\-]\s*(.+?)\s*$",
            r"^(?:option|choice)\s+([A-Z])\s*[:\-]?\s*(.+?)\s*$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper(), str(match.group(2) or "").strip()
        return None, text

    def _normalize_choice_match_text(self, text: str) -> str:
        _, body = self._split_answer_choice_label(text)
        normalized = " ".join(str(body or "").strip().lower().split())
        return normalized.strip(" \t\r\n.,;:()[]{}")

    def _normalize_numeric_expression(self, text: str) -> str:
        expr = str(text or "").strip()
        if not expr:
            return ""

        replacements = {
            "−": "-",
            "–": "-",
            "—": "-",
            "×": "*",
            "÷": "/",
            "π": "pi",
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)

        prev = None
        while expr != prev:
            prev = expr
            expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
            expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)

        expr = re.sub(r"(?i)\bsqrt\s+(\d+(?:\.\d+)?)\b", r"sqrt(\1)", expr)
        expr = expr.replace("^", "**")
        return expr.strip("`$ ")

    def _safe_numeric_eval_ast(self, node) -> float:
        if isinstance(node, ast.Expression):
            return self._safe_numeric_eval_ast(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._safe_numeric_eval_ast(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = self._safe_numeric_eval_ast(node.left)
            right = self._safe_numeric_eval_ast(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left ** right
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return float(math.pi)
            if node.id == "e":
                return float(math.e)
            raise ValueError("Unsupported symbol")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = [self._safe_numeric_eval_ast(arg) for arg in node.args]
            if name == "sqrt" and len(args) == 1:
                return float(math.sqrt(args[0]))
            if name == "abs" and len(args) == 1:
                return float(abs(args[0]))
            raise ValueError("Unsupported function")
        raise ValueError("Unsupported expression")

    def _extract_numeric_value(self, text: str):
        expr = self._normalize_numeric_expression(text)
        if not expr:
            return None

        candidates = []

        def _add_candidate(value):
            value = str(value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        _add_candidate(expr)
        for separator in ("=", "≈", "~"):
            if separator in expr:
                for part in reversed(expr.split(separator)):
                    _add_candidate(part.strip(" \t\r\n,;:"))
        for part in re.split(r"(?i)\b(?:approximately|approx\.?|about)\b", expr):
            _add_candidate(part.strip(" \t\r\n,;:"))
        for token in re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", expr):
            _add_candidate(token)

        for candidate in candidates:
            try:
                value = self._safe_numeric_eval_ast(ast.parse(candidate, mode="eval"))
            except Exception:
                continue
            if math.isfinite(value):
                return float(value)
        return None

    def _coerce_answer_choice_hint(self, answer_choice: str | None, choices: list[str]) -> str | None:
        normalized_hint = " ".join(str(answer_choice or "").strip().lower().split())
        if not normalized_hint:
            return None
        for choice in choices or []:
            original = str(choice or "").strip()
            label, body = self._split_answer_choice_label(original)
            variants = {
                " ".join(original.lower().split()),
                " ".join(str(body or "").lower().split()),
            }
            if label:
                variants.update({label.lower(), f"option {label.lower()}", f"choice {label.lower()}"})
            if normalized_hint in variants:
                return original
        return None

    def _match_math_solver_answer_choice(
        self,
        result_value: str | None,
        choices: list[str],
        hinted_choice: str | None = None,
        insufficient_information: bool = False,
    ) -> str | None:
        if insufficient_information or not choices:
            return None

        hinted = self._coerce_answer_choice_hint(hinted_choice, choices)
        if hinted:
            return hinted

        target_text = str(result_value or "").strip()
        if not target_text:
            return None

        normalized_target = self._normalize_choice_match_text(target_text)
        if not normalized_target:
            return None

        prepared = []
        for choice in choices:
            original = str(choice or "").strip()
            if not original:
                continue
            label, body = self._split_answer_choice_label(original)
            prepared.append(
                {
                    "original": original,
                    "label": label,
                    "body": body,
                    "normalized_original": " ".join(original.lower().split()),
                    "normalized_body": " ".join(str(body or "").lower().split()),
                    "numeric_value": self._extract_numeric_value(body or original),
                }
            )

        for choice in prepared:
            if normalized_target in {choice["normalized_original"], choice["normalized_body"]}:
                return choice["original"]
            if choice["label"] and normalized_target in {
                choice["label"].lower(),
                f"option {choice['label'].lower()}",
                f"choice {choice['label'].lower()}",
            }:
                return choice["original"]

        target_value = self._extract_numeric_value(target_text)
        if target_value is not None:
            best_choice = None
            best_diff = None
            for choice in prepared:
                numeric_value = choice["numeric_value"]
                if numeric_value is None:
                    continue
                diff = abs(float(numeric_value) - float(target_value))
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_choice = choice["original"]
            if best_choice is not None:
                return best_choice

        for choice in prepared:
            normalized_body = choice["normalized_body"]
            if normalized_body and (
                normalized_body in normalized_target or normalized_target in normalized_body
            ):
                return choice["original"]
        return None

    def _select_chart_analysis_frame_result(self, frame_results: list[dict]) -> dict:
        if not frame_results:
            return {}

        ranked = sorted(
            frame_results,
            key=lambda item: (
                -float(item.get("score", -1.0) or -1.0),
                float(item.get("timestamp", 0.0) or 0.0),
            ),
        )
        best_frame = ranked[0]
        result = dict(best_frame.get("result") or {})
        result["selected_frame"] = {
            "frame_path": best_frame.get("frame_path"),
            "timestamp": best_frame.get("timestamp"),
            "score": best_frame.get("score"),
        }
        result["frame_results"] = frame_results
        result["multi_frame_strategy"] = "best_frame"
        result["selection_reason"] = (
            "Top-level fields mirror one high-salience candidate frame; inspect "
            "frame_results for all frame-level chart or diagram reads."
        )
        return result

    def _geometry_evidence_conflicts(self, evidence) -> list[str]:
        if isinstance(evidence, list):
            text = "\n".join(str(item or "") for item in evidence)
        else:
            text = str(evidence or "")
        normalized = self._normalize_chart_text(text)
        conflicts = []

        has_negative_tangency = (
            "not tangent" in normalized
            or "not by tangency" in normalized
            or "not tangent to any of the inner arcs" in normalized
        )
        has_positive_tangency = (
            " is tangent to " in f" {normalized} "
            or " are tangent to " in f" {normalized} "
            or " tangent at " in normalized
            or " tangency to " in normalized
        )
        if has_negative_tangency and has_positive_tangency:
            conflicts.append("The evidence does not consistently establish which curves are tangent.")

        has_negative_semicircle = "not a full semicircle" in normalized
        has_positive_semicircle = (
            "is a semicircle" in normalized
            or "are semicircles" in normalized
            or "each curved arc is a semicircle" in normalized
            or "each arc is a semicircle" in normalized
        )
        if has_negative_semicircle and has_positive_semicircle:
            conflicts.append("The evidence does not consistently establish whether the curved piece is a full semicircle or a smaller arc.")

        determined_by_vertices = (
            "defined by the four vertices" in normalized
            or "determined by the square's vertices" in normalized
            or "passes through the four vertices" in normalized
        )
        determined_by_tangency = (
            "determined by tangency" in normalized
            or "by tangency to the inner arcs" in normalized
        )
        if determined_by_vertices and determined_by_tangency:
            conflicts.append("The evidence gives incompatible rules for what determines the larger boundary.")

        return conflicts

    def _enforce_math_solver_consistency(self, evidence, result: dict) -> dict:
        if not isinstance(result, dict):
            return result

        conflicts = self._geometry_evidence_conflicts(evidence)
        if not conflicts or bool(result.get("insufficient_information")):
            return result

        enforced = dict(result)
        enforced["insufficient_information"] = True
        enforced["result"] = None
        enforced["answer_choice"] = None
        try:
            enforced["confidence"] = min(float(enforced.get("confidence", 0.0) or 0.0), 0.35)
        except (TypeError, ValueError):
            enforced["confidence"] = 0.35

        missing_facts = [
            str(item or "").strip()
            for item in (enforced.get("missing_facts") or [])
            if str(item or "").strip()
        ]
        for conflict in conflicts:
            if conflict not in missing_facts:
                missing_facts.append(conflict)
        enforced["missing_facts"] = missing_facts

        derivation = [
            str(item or "").strip()
            for item in (enforced.get("derivation") or [])
            if str(item or "").strip()
        ]
        note = "The provided evidence contains unresolved conflicting geometry descriptions, so a unique derivation is not justified."
        if note not in derivation:
            derivation.append(note)
        enforced["derivation"] = derivation
        return enforced

    def _resolve_frame_bundle(self, arguments: dict):
        raw_paths = arguments.get("frame_paths", arguments.get("frame_path"))
        raw_ts = arguments.get("timestamps", arguments.get("timestamp"))

        frame_paths = []
        timestamps = []

        if isinstance(raw_paths, list):
            for item in raw_paths:
                if isinstance(item, dict):
                    frame_paths.append(item.get("frame_path"))
                    timestamps.append(item.get("timestamp"))
                else:
                    frame_paths.append(item)
        elif raw_paths is not None:
            frame_paths = [raw_paths]

        if isinstance(raw_ts, str):
            parsed = robust_eval(raw_ts)
            raw_ts = parsed if isinstance(parsed, list) else [raw_ts]
        elif raw_ts is None:
            raw_ts = []
        elif not isinstance(raw_ts, list):
            raw_ts = [raw_ts]

        if raw_ts:
            timestamps = list(raw_ts)

        resolved_paths = []
        resolved_timestamps = []
        total = max(len(frame_paths), len(timestamps))

        for i in range(total):
            frame_path = frame_paths[i] if i < len(frame_paths) else None
            frame_ts = timestamps[i] if i < len(timestamps) else None
            frame_ts = self._safe_float(frame_ts, None)

            if not frame_path or not os.path.exists(frame_path):
                if frame_ts is None:
                    continue
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)
            else:
                if frame_ts is None:
                    try:
                        frame_ts = float(self._timestamp_from_dense_frame_path(frame_path))
                    except Exception:
                        frame_ts = 0.0

            if frame_path and os.path.exists(frame_path):
                resolved_paths.append(frame_path)
                resolved_timestamps.append(float(frame_ts))

        if len(resolved_paths) > 1:
            ordered = sorted(zip(resolved_timestamps, resolved_paths), key=lambda item: item[0])
            deduped = []
            last_ts = None
            for ts, path in ordered:
                if last_ts is not None and ts <= last_ts:
                    continue
                deduped.append((ts, path))
                last_ts = ts
            resolved_timestamps = [ts for ts, _ in deduped]
            resolved_paths = [path for _, path in deduped]

        return resolved_paths, resolved_timestamps

    def _latest_prior_step_of_type(self, before_step: int, step_tools: dict, target_tool: str):
        candidates = [
            step for step, tool in step_tools.items()
            if step < before_step and tool == target_tool
        ]
        return max(candidates) if candidates else None

    def _select_frames_aligned_with_temporal_grounder(
        self,
        frame_result: dict,
        temporal_result: dict,
    ) -> list:
        frames = frame_result.get("frames") or []
        segments = temporal_result.get("segments") or []
        if not isinstance(frames, list) or not isinstance(segments, list):
            return []

        normalized_frames = []
        for item in frames:
            if not isinstance(item, dict):
                continue
            frame_path = item.get("frame_path")
            ts = self._safe_float(item.get("timestamp"), None)
            if not frame_path or ts is None:
                continue
            normalized_frames.append(
                {
                    "frame_path": frame_path,
                    "timestamp": float(ts),
                    "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                }
            )
        if not normalized_frames:
            return []

        ranked_segments = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = self._safe_float(seg.get("start"), None)
            end = self._safe_float(seg.get("end"), None)
            if start is None or end is None:
                continue
            conf = float(seg.get("confidence", 0.0) or 0.0)
            ranked_segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "confidence": conf,
                    "mid": (float(start) + float(end)) / 2.0,
                }
            )
        if not ranked_segments:
            return []

        ranked_segments.sort(key=lambda seg: seg["confidence"], reverse=True)
        tol = max(0.0, float(getattr(self, "dense_segment_half_width", 0.5)))

        # Only consider the top-1 reranked segment (highest confidence).
        top_seg = ranked_segments[0]
        aligned = [
            frame for frame in normalized_frames
            if top_seg["start"] - tol <= frame["timestamp"] <= top_seg["end"] + tol
        ]
        if aligned:
            return sorted(
                aligned,
                key=lambda frame: (
                    -frame["relevance_score"],
                    abs(frame["timestamp"] - top_seg["mid"]),
                    frame["timestamp"],
                ),
            )

        # No frames fell within the top segment window — return all retrieved
        # frames rather than discarding all but the nearest one.  The downstream tool
        # (chart_analyzer, OCR, etc.) will evaluate every candidate and pick the best.
        return sorted(
            normalized_frames,
            key=lambda frame: (-frame["relevance_score"], frame["timestamp"]),
        )

    def _extract_retrieved_frames(self, frame_result: dict) -> list:
        frames = frame_result.get("frames") or []
        if not isinstance(frames, list):
            return []

        normalized = []
        for item in frames:
            if not isinstance(item, dict):
                continue
            frame_path = item.get("frame_path")
            ts = self._safe_float(item.get("timestamp"), None)
            if not frame_path or ts is None:
                continue
            normalized.append(
                {
                    "frame_path": frame_path,
                    "timestamp": float(ts),
                    "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                }
            )
        return sorted(normalized, key=lambda item: (item["timestamp"], -item["relevance_score"]))

    def _align_visual_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
        current_step: int,
        depends_on: list,
        step_results: dict,
        step_tools: dict,
        fallback_step_results=None,
        fallback_step_tools=None,
    ) -> dict:
        fallback_step_results = fallback_step_results or {}
        fallback_step_tools = fallback_step_tools or {}
        current_plan_steps = set(step_tools.keys())

        def _get_dep_result(dep: int, expected_tool: str):
            if step_tools.get(dep) == expected_tool:
                result = step_results.get(dep)
                return dep, result if isinstance(result, dict) else None
            if dep in current_plan_steps:
                return dep, None
            if fallback_step_tools.get(dep) == expected_tool:
                result = fallback_step_results.get(dep)
                return dep, result if isinstance(result, dict) else None
            return dep, None

        if tool_name == "frame_retriever":
            temporal_step = None
            temporal_result = None
            for dep in depends_on:
                dep_step, dep_result = _get_dep_result(dep, "temporal_grounder")
                if dep_result is not None:
                    temporal_step = dep_step
                    temporal_result = dep_result
            if temporal_step is None:
                return arguments

            temporal_result = temporal_result or {}
            segments = [
                seg for seg in (temporal_result.get("segments") or [])
                if isinstance(seg, dict)
                and self._safe_float(seg.get("start"), None) is not None
                and self._safe_float(seg.get("end"), None) is not None
            ]
            if not segments:
                return arguments

            ranked_segments = sorted(
                segments,
                key=lambda seg: float(seg.get("confidence", 0.0) or 0.0),
                reverse=True,
            )
            # Snap generated timestamps to the nearest dense-frame grid so they
            # hit the pre-built embedding cache instead of triggering fresh
            # per-frame forward passes.  dense_frame_fps is typically 1.0, so
            # the grid is integer seconds; at higher fps it's finer.
            dense_fps = max(0.1, float(getattr(self, "dense_frame_fps", 1.0)))
            snap_interval = 1.0 / dense_fps

            def _snap(t: float) -> float:
                return round(round(t / snap_interval) * snap_interval, 2)

            timestamps = []
            for seg in ranked_segments:
                start = float(seg["start"])
                end = float(seg["end"])
                if end <= start:
                    timestamps.append(_snap(start))
                    continue
                step = (end - start) / 4.0
                timestamps.extend([
                    _snap(start + step),
                    _snap(start + 2.0 * step),
                    _snap(start + 3.0 * step),
                ])

            # Also record the bounding time_range across ALL grounded segments so
            # the query-only retrieval path (_informative_retrieval) is similarly scoped.
            all_starts = [float(seg["start"]) for seg in segments]
            all_ends = [float(seg["end"]) for seg in segments]
            time_range = [min(all_starts), max(all_ends)]

            updated = dict(arguments)
            updated["timestamps"] = timestamps
            updated["time_range"] = time_range
            return updated

        if tool_name not in {"spatial_grounder", "counter", "chart_analyzer", "ocr"}:
            return arguments

        frame_step = None
        frame_result = None
        for dep in depends_on:
            dep_step, dep_result = _get_dep_result(dep, "frame_retriever")
            if dep_result is not None:
                frame_step = dep_step
                frame_result = dep_result
        if frame_step is None:
            return arguments

        frame_result = frame_result or {}
        retrieved_frames = self._extract_retrieved_frames(frame_result)
        if not retrieved_frames:
            return arguments

        temporal_step = None
        temporal_result = None
        for dep in depends_on:
            dep_step, dep_result = _get_dep_result(dep, "temporal_grounder")
            if dep_result is not None:
                temporal_step = dep_step
                temporal_result = dep_result
        if temporal_step is not None:
            aligned_frames = self._select_frames_aligned_with_temporal_grounder(
                frame_result,
                temporal_result or {},
            )
            if not aligned_frames:
                aligned_frames = retrieved_frames
            else:
                # Cap to 10 frames from the top-ranked clip
                aligned_frames = aligned_frames[:10]
        else:
            aligned_frames = retrieved_frames

        updated = dict(arguments)
        if tool_name in {"chart_analyzer", "ocr"}:
            updated["frame_path"] = [
                {"frame_path": item["frame_path"], "timestamp": item["timestamp"]}
                for item in aligned_frames
            ]
            updated["timestamp"] = None
        else:
            updated["frame_path"] = [
                {"frame_path": item["frame_path"], "timestamp": item["timestamp"]}
                for item in aligned_frames
            ]
            updated["timestamp"] = None
        return updated

    def _load_chart_model(self):
        if getattr(self, "_chart_model", None) is not None:
            return
        device = self._chart_torch_device()
        model_name = self.chart_model_name
        print(f"  Loading chart VLM: {model_name} on {device}")
        self._chart_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)
        self._chart_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        print(f"  ✓ Chart VLM loaded: {model_name}")

    def _call_chart_vision_api(self, prompt_text: str, frame_paths) -> str:
        """Call the chart vision API with one or more frame images."""
        if isinstance(frame_paths, str):
            frame_paths = [frame_paths]

        def _encode(frame_path: str) -> str:
            img = Image.open(frame_path).convert("RGB")
            width, height = img.size
            if max(width, height) > 768:
                if width > height:
                    nw, nh = 768, int(height * (768 / width))
                else:
                    nh, nw = 768, int(width * (768 / height))
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        content = []
        for fp in frame_paths:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(fp)}"}})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        vlm_out = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
        if vlm_out:
            refiner_debug.write_json(
                vlm_out,
                "model_input.json",
                {"model": self.chart_model_name, "messages": messages, "frame_paths": frame_paths},
            )
        bases = self._chart_effective_api_bases()
        keys = self._chart_effective_api_keys()
        pairs = list(zip(bases, keys))
        if not pairs:
            print("[CHART_VISION_API] ERROR: no API base/key; set chart_api_base/chart_api_keys or planner_api_* for api mode")
            return ""
        for base, key in pairs:
            try:
                client = OpenAI(base_url=base.strip(), api_key=key.strip())
                completion = client.chat.completions.create(
                    model=self.chart_model_name,
                    messages=messages,
                    **openai_chat_temperature_kwargs(self.chart_model_name, 0.0),
                    **openai_chat_completion_limit_kwargs(self.chart_model_name, 1024),
                )
                out = completion.choices[0].message.content
                text = out if isinstance(out, str) else (out or "")
                if vlm_out:
                    refiner_debug.write_text(vlm_out, "vlm_raw_output.txt", text)
                return text
            except Exception as e:
                print(f"[CHART_VISION_API] ERROR base={base} model={self.chart_model_name}: {e}")
        return ""

    def _spatial_grounder_torch_device(self):
        raw = str(getattr(self, "spatial_grounder_device", "cuda:0")).strip()
        if not raw or raw.lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if raw.isdigit():
            raw = f"cuda:{int(raw)}"
        try:
            device = torch.device(raw)
        except Exception:
            fallback = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"  Spatial grounder device {raw!r} is invalid; falling back to {fallback}.")
            return fallback
        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("  Spatial grounder requested CUDA, but CUDA is unavailable; falling back to cpu.")
                return torch.device("cpu")
            if device.index is not None and device.index >= torch.cuda.device_count():
                fallback = torch.device("cuda:0")
                print(f"  Spatial grounder device {raw!r} is out of range; falling back to {fallback}.")
                return fallback
        return device

    def _load_grounding_dino_model(self):
        model_name = str(
            getattr(self, "spatial_grounder_model_name", "IDEA-Research/grounding-dino-base")
            or "IDEA-Research/grounding-dino-base"
        ).strip()
        device = self._spatial_grounder_torch_device()
        loaded_name = getattr(self, "_grounding_dino_loaded_model_name", None)
        loaded_device = getattr(self, "_grounding_dino_loaded_device", None)
        if (
            getattr(self, "_grounding_dino_model", None) is not None
            and getattr(self, "_grounding_dino_processor", None) is not None
            and loaded_name == model_name
            and loaded_device == str(device)
        ):
            return self._grounding_dino_processor, self._grounding_dino_model, device

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as e:
            raise RuntimeError(f"transformers Grounding DINO classes are unavailable: {e}") from e

        print(f"  Loading spatial grounder: {model_name} on {device}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
        ).eval().to(device)

        self._grounding_dino_processor = processor
        self._grounding_dino_model = model
        self._grounding_dino_loaded_model_name = model_name
        self._grounding_dino_loaded_device = str(device)
        print(f"  ✓ Spatial grounder loaded: {model_name}")
        return processor, model, device

    def _grounding_dino_model_dtype(self, model) -> torch.dtype:
        dtype = getattr(model, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            return next(model.parameters()).dtype
        except (StopIteration, AttributeError, TypeError):
            return torch.float32

    def _normalize_grounding_dino_query(self, query: str) -> str:
        text = " ".join(str(query or "").strip().split()).lower()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _bbox_area_fraction(self, bbox, width: int, height: int) -> float:
        if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
            return 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return round(area / float(width * height), 4)

    def _bbox_region(self, bbox, width: int, height: int) -> str:
        if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
            return "unknown"
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = ((x1 + x2) / 2.0) / float(width)
        cy = ((y1 + y2) / 2.0) / float(height)
        horizontal = "left" if cx < (1.0 / 3.0) else ("center" if cx < (2.0 / 3.0) else "right")
        vertical = "top" if cy < (1.0 / 3.0) else ("middle" if cy < (2.0 / 3.0) else "bottom")
        if horizontal == "center" and vertical == "middle":
            return "center"
        return f"{horizontal}-{vertical}"

    def _spatial_description_from_detections(self, detections: list, width: int, height: int) -> str:
        if not detections:
            return ""
        phrases = []
        for det in detections[:5]:
            if not isinstance(det, dict):
                continue
            label = str(det.get("label") or "object").strip()
            bbox = det.get("bbox") or []
            if len(bbox) != 4:
                continue
            phrases.append(f"{label} at {self._bbox_region(bbox, width, height)}")
        if not phrases:
            return ""
        count = len([d for d in detections if isinstance(d, dict)])
        noun = "detection" if count == 1 else "detections"
        return f"{count} {noun}: " + "; ".join(phrases)

    def _dedupe_grounding_dino_detections(self, detections: list, iou_threshold: float) -> list:
        if len(detections) < 2:
            return detections
        try:
            from torchvision.ops import nms
        except Exception:
            return detections

        thr = float(self._safe_float(iou_threshold, 0.8) or 0.8)
        thr = max(0.0, min(1.0, thr))
        keep_indices = []
        labels = sorted({str(det.get("label") or "") for det in detections if isinstance(det, dict)})
        for label in labels:
            label_indices = [
                idx for idx, det in enumerate(detections)
                if isinstance(det, dict) and str(det.get("label") or "") == label
            ]
            if len(label_indices) <= 1:
                keep_indices.extend(label_indices)
                continue
            boxes = torch.tensor(
                [detections[idx]["bbox"] for idx in label_indices],
                dtype=torch.float32,
            )
            scores = torch.tensor(
                [float(detections[idx].get("confidence", 0.0) or 0.0) for idx in label_indices],
                dtype=torch.float32,
            )
            kept = nms(boxes, scores, thr).tolist()
            keep_indices.extend(label_indices[i] for i in kept)

        keep_indices = sorted(
            set(keep_indices),
            key=lambda idx: float(detections[idx].get("confidence", 0.0) or 0.0),
            reverse=True,
        )
        return [detections[idx] for idx in keep_indices]

    def _run_spatial_grounder_vlm(self, query: str, frame_path: str, frame_ts: float, return_masks: bool = False):
        default_result = {
            "query": query,
            "detections": [],
            "spatial_description": "",
            "backend": "vlm",
        }
        prompt = (
            spatial_grunder_prompt.strip()
            + f"\n\nQuery: {query}\nreturn_masks: {str(bool(return_masks)).lower()}\n"
            + "Return JSON only matching the OUTPUT FORMAT above.\n"
        )
        result = self._run_vlm_json(
            prompt,
            [frame_path] if frame_path else [],
            [float(self._safe_float(frame_ts, 0.0) or 0.0)],
            default_result,
        )
        if isinstance(result, dict):
            result.setdefault("query", query)
            result.setdefault("detections", [])
            result.setdefault("spatial_description", "")
            result.setdefault("backend", "vlm")
        return result

    def _run_grounding_dino_spatial_grounder(self, frame_path: str, query: str, return_masks: bool = False):
        if not frame_path or not os.path.exists(frame_path):
            return None, "Grounding DINO spatial grounding skipped because the frame path was unavailable."

        model_name = str(
            getattr(self, "spatial_grounder_model_name", "IDEA-Research/grounding-dino-base")
            or "IDEA-Research/grounding-dino-base"
        ).strip()
        prompt_text = self._normalize_grounding_dino_query(query)
        if not prompt_text:
            return None, "Grounding DINO spatial grounding skipped because the query was empty."

        try:
            processor, model, device = self._load_grounding_dino_model()
        except Exception as e:
            return None, f"Grounding DINO backend unavailable because the Hugging Face model could not be loaded: {e}"
        model_dtype = self._grounding_dino_model_dtype(model)

        box_threshold = float(getattr(self, "spatial_grounder_box_threshold", 0.25))
        iou_threshold = float(getattr(self, "spatial_grounder_iou_threshold", 0.8))
        text_threshold = float(
            self._safe_float(
                getattr(self, "spatial_grounder_text_threshold", box_threshold),
                box_threshold,
            ) or box_threshold
        )

        debug_dir = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
        request_payload = {
            "model": model_name,
            "device": str(device),
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "query": prompt_text,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "iou_threshold": iou_threshold,
            "return_masks": bool(return_masks),
            "frame_path": frame_path,
        }
        if debug_dir:
            refiner_debug.write_json(debug_dir, "grounding_dino_request.json", request_payload)

        try:
            with Image.open(frame_path) as img:
                image = img.convert("RGB")
                width, height = image.size
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            model_inputs = {}
            for key, value in inputs.items():
                if not isinstance(value, torch.Tensor):
                    model_inputs[key] = value
                    continue
                moved = value.to(device)
                if moved.is_floating_point():
                    moved = moved.to(dtype=model_dtype)
                model_inputs[key] = moved
            with torch.inference_mode():
                outputs = model(**model_inputs)
            hf_results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)],
            )
        except Exception as e:
            return None, f"Grounding DINO inference failed: {e}"

        raw_result = {}
        if hf_results:
            first = hf_results[0]
            score_values = first.get("scores")
            label_values = first.get("labels")
            box_values = first.get("boxes")
            if hasattr(score_values, "tolist"):
                score_values = score_values.tolist()
            elif score_values is None:
                score_values = []
            else:
                score_values = list(score_values)
            if hasattr(label_values, "tolist"):
                label_values = label_values.tolist()
            elif label_values is None:
                label_values = []
            else:
                label_values = list(label_values)
            if hasattr(box_values, "tolist"):
                box_values = box_values.tolist()
            elif box_values is None:
                box_values = []
            else:
                box_values = [
                    box.tolist() if hasattr(box, "tolist") else list(box)
                    for box in box_values
                ]
            raw_result = {
                "scores": [round(float(score), 4) for score in score_values],
                "labels": [str(label) for label in label_values],
                "boxes": [
                    [round(float(v), 2) for v in box]
                    for box in box_values
                ],
            }
        if debug_dir:
            refiner_debug.write_json(debug_dir, "grounding_dino_raw_response.json", raw_result)

        detections = []
        scores = raw_result.get("scores") if isinstance(raw_result, dict) else None
        labels = raw_result.get("labels") if isinstance(raw_result, dict) else None
        boxes = raw_result.get("boxes") if isinstance(raw_result, dict) else None
        total = min(len(scores or []), len(labels or []), len(boxes or []))
        for idx in range(total):
            bbox = boxes[idx]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            box = [int(round(float(self._safe_float(v, 0.0) or 0.0))) for v in bbox]
            score = float(self._safe_float(scores[idx], 0.0) or 0.0)
            detections.append(
                {
                    "label": str(labels[idx] or query).strip() or query,
                    "bbox": box,
                    "confidence": max(0.0, min(1.0, score)),
                    "mask_path": None,
                    "area_fraction": self._bbox_area_fraction(box, width, height),
                }
            )

        detections.sort(key=lambda det: float(det.get("confidence", 0.0) or 0.0), reverse=True)
        detections = self._dedupe_grounding_dino_detections(detections, iou_threshold)
        result = {
            "query": query,
            "detections": detections,
            "spatial_description": self._spatial_description_from_detections(detections, width, height),
            "backend": "grounding_dino_hf",
            "grounding_model": model_name,
            "grounding_device": str(device),
        }
        if return_masks:
            result["warning"] = (
                "Grounding DINO mask outputs are requested, but the Hugging Face runner currently returns bbox detections only; "
                "mask_path remains null."
            )
        return result, None

    def _informative_retrieval(self, query: str, top_k: int, time_range: tuple = None):
        """Dense-frame text–image retrieval by default; optional clip-level search.
        time_range: optional (start_sec, end_sec) to restrict search to a temporal window.
        """
        if getattr(self, "use_clip_retrieval", False):
            return "segment", self._informative_clip_retrieval(query, top_k)

        frame_paths, _ = self._list_dense_frame_paths(self.dataset_folder, self.video_path)
        if not frame_paths and self.duration is not None and float(self.duration) > 0:
            try:
                from video_utils import timestamp_to_clip_path
                timestamp_to_clip_path(
                    self.dataset_folder, 0.0, float(self.duration), self.video_path,
                    fps=float(getattr(self, "dense_frame_fps", 24.0)),
                )
            except Exception as e:
                print(f"  dense frame materialize skipped: {e}")
            frame_paths, _ = self._list_dense_frame_paths(self.dataset_folder, self.video_path)
        if not frame_paths:
            return "dense", []
        frame_items = [
            {
                "frame_path": fp,
                "timestamp": self._timestamp_from_dense_frame_path(fp),
            }
            for fp in frame_paths
        ]
        if time_range is not None:
            t_start, t_end = float(time_range[0]), float(time_range[1])
            frame_items = [f for f in frame_items if t_start <= f["timestamp"] <= t_end]
            if not frame_items:
                # If the filter removed everything (rounding edge), fall back to full set
                frame_items = [
                    {"frame_path": fp, "timestamp": self._timestamp_from_dense_frame_path(fp)}
                    for fp in frame_paths
                ]
        try:
            scored = self._qwen_score_frames(query, frame_items, top_k)
            return "dense", [(item["frame_path"], item["relevance_score"]) for item in scored]
        except Exception as e:
            print(f"  qwen dense retrieval error: {e}")
            return "dense", []

    def _dense_frame_embed_cache_paths(self):
        """Returns (embeddings_pt_path, frame_paths_json_path) for the current video."""
        from pathlib import Path as _Path
        video_id = _Path(str(self.video_path)).stem
        cache_dir = _Path(self.dataset_folder) / "dense_frames" / video_id
        return cache_dir / "qwen_frame_embeddings.pt", cache_dir / "qwen_frame_paths.json"

    def _temporal_grounder_clip_embed_cache_paths(self):
        """Returns (embeddings_pt_path, windows_json_path) for temporal-grounder clips."""
        from pathlib import Path as _Path
        video_id = _Path(str(self.video_path)).stem
        cache_dir = _Path(self.dataset_folder) / "temporal_grounder_cache" / video_id
        return cache_dir / "qwen_clip_embeddings.pt", cache_dir / "qwen_clip_windows.json"

    def _load_temporal_grounder_clip_embeddings_cache(self, cache_meta: dict):
        cached = getattr(self, "_temporal_grounder_qwen_clip_embeddings_cache", None)
        if isinstance(cached, dict) and all(cached.get(k) == v for k, v in cache_meta.items()):
            cached_windows = cached.get("clips")
            cached_embeddings = cached.get("clip_embeddings")
            if isinstance(cached_windows, list) and isinstance(cached_embeddings, torch.Tensor):
                return list(cached_windows), cached_embeddings.float()

        emb_path, windows_path = self._temporal_grounder_clip_embed_cache_paths()
        if not (emb_path.is_file() and windows_path.is_file()):
            return None, None

        try:
            payload = json.loads(windows_path.read_text(encoding="utf-8"))
            saved_meta = payload.get("cache_meta") or {}
            saved_windows = payload.get("clips") or []
            saved_embeddings = torch.load(emb_path, map_location="cpu").float()
            if (
                isinstance(saved_windows, list)
                and isinstance(saved_embeddings, torch.Tensor)
                and all(saved_meta.get(k) == v for k, v in cache_meta.items())
                and int(saved_embeddings.shape[0]) == len(saved_windows)
            ):
                self._temporal_grounder_qwen_clip_embeddings_cache = {
                    **cache_meta,
                    "clips": list(saved_windows),
                    "clip_embeddings": saved_embeddings,
                }
                return list(saved_windows), saved_embeddings
        except Exception as e:
            print(f"  [temporal grounder cache] load failed, recomputing: {e}")

        return None, None

    def _save_temporal_grounder_clip_embeddings_cache(
        self,
        cache_meta: dict,
        windows: list,
        clip_embeddings: torch.Tensor,
    ):
        emb_path, windows_path = self._temporal_grounder_clip_embed_cache_paths()
        clip_embeddings = clip_embeddings.detach().cpu().float()
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clip_embeddings, emb_path)
        windows_path.write_text(
            json.dumps({"cache_meta": cache_meta, "clips": list(windows)}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._temporal_grounder_qwen_clip_embeddings_cache = {
            **cache_meta,
            "clips": list(windows),
            "clip_embeddings": clip_embeddings,
        }

    def _get_or_load_frame_embedder(self):
        """Return the resident Qwen3-VL-Embedding-8B instance, loading it on first call.
        Stays on GPU 1 (FRAME_EMBEDDER_DEVICE_INDEX) for the entire pipeline so that
        only the query (1 forward pass) needs to run per frame_retriever call.
        Call _release_frame_embedder() only at pipeline end to free GPU memory.
        """
        if getattr(self, "_frame_embedder", None) is not None:
            return self._frame_embedder

        model_dir = self._resolve_temporal_grounder_model_dir(
            getattr(self, "temporal_grounder_model_name", "Qwen/Qwen3-VL-Embedding-8B")
        )
        Embedder = self._load_temporal_grounder_helper_class(
            model_dir, "qwen3_vl_embedding.py", "Qwen3VLEmbedder",
            "_temporal_grounder_embedder_class",
        )
        model_kwargs = self._temporal_grounder_model_kwargs()
        embed_dev_idx = self._frame_embedder_device_index()
        print(f"  [frame embedder] Loading Qwen3-VL-Embedding-8B onto GPU {embed_dev_idx} (resident) ...")
        with torch.cuda.device(embed_dev_idx):
            self._frame_embedder = Embedder(model_name_or_path=str(model_dir), **model_kwargs)
        print(f"  [frame embedder] Loaded — resident on GPU {embed_dev_idx} for the full pipeline.")
        return self._frame_embedder

    def _release_frame_embedder(self):
        """Release the resident frame embedder from GPU memory (called at pipeline end)."""
        embedder = getattr(self, "_frame_embedder", None)
        if embedder is not None:
            embed_dev_idx = self._frame_embedder_device_index()
            model = getattr(embedder, "model", None)
            if isinstance(model, torch.nn.Module):
                try:
                    model.cpu()
                except Exception:
                    pass
            for attr in ("model", "processor", "tokenizer", "score_linear"):
                if hasattr(embedder, attr):
                    try:
                        setattr(embedder, attr, None)
                    except Exception:
                        pass
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                with torch.cuda.device(embed_dev_idx):
                    torch.cuda.empty_cache()
            self._frame_embedder = None
            print("  [frame embedder] Released from GPU.")

    def _precompute_frame_embeddings_cache(self, frame_items: list) -> bool:
        """Embed *all* frame_items with Qwen3-VL-Embedding-8B and save to disk.
        Idempotent — skips when cache already exists.  Returns True on success.
        Reuses the resident frame embedder so load cost is paid at most once."""
        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        if emb_path.is_file() and paths_path.is_file():
            return True
        if not frame_items:
            return False

        def _as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float()
            return torch.tensor(x, dtype=torch.float32)

        batch_size = max(1, int(getattr(self, "dense_frame_embed_batch", 8)))
        print(f"  [frame embed cache] Precomputing embeddings for {len(frame_items)} frames ...")
        embedder = self._get_or_load_frame_embedder()
        try:
            all_embs = []
            total_batches = (len(frame_items) + batch_size - 1) // batch_size
            for i in tqdm(
                range(0, len(frame_items), batch_size),
                total=total_batches,
                desc="Qwen frame embedding",
                unit="batch",
            ):
                batch = frame_items[i : i + batch_size]
                samples = [{"video": [item["frame_path"]]} for item in batch]
                try:
                    with self._frame_embedder_inference_context():
                        embs = _as_tensor(embedder.process(samples))
                except Exception:
                    embs_list = []
                    for item in batch:
                        try:
                            with self._frame_embedder_inference_context():
                                e = _as_tensor(embedder.process([{"video": [item["frame_path"]]}]))
                        except Exception:
                            e = torch.zeros(1, 4096)
                        embs_list.append(e)
                    embs = torch.cat(embs_list, dim=0)
                embs_norm = embs / embs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                all_embs.append(embs_norm.cpu())

            frame_embs = torch.cat(all_embs, dim=0).float()
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(frame_embs, emb_path)
            paths_path.write_text(
                json.dumps([item["frame_path"] for item in frame_items]),
                encoding="utf-8",
            )
            print(f"  [frame embed cache] Saved {frame_embs.shape[0]} frame embeddings → {emb_path}")
            return True
        except Exception as e:
            print(f"  [frame embed cache] Precompute failed: {e}")
            return False
        # NOTE: embedder is intentionally NOT released here — it stays resident

    def _qwen_score_frames(self, query: str, frame_items: list, top_k: int) -> list:
        """Score frame_items against query using Qwen3-VL-Embedding-8B.
        Frame embeddings are served from a disk cache built on first call — only
        the query is embedded live, reducing cost from O(N_frames) to O(1) FWD passes.
        Returns a list sorted by relevance_score descending, capped at top_k.
        Each output item: {frame_path, timestamp, relevance_score}.
        """
        def _as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float()
            return torch.tensor(x, dtype=torch.float32)

        batch_size = max(1, int(getattr(self, "dense_frame_embed_batch", 8)))

        # ── Load cached frame embeddings ──────────────────────────────────────
        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        cached_frame_embs: torch.Tensor | None = None
        cached_paths_index: dict = {}
        if emb_path.is_file() and paths_path.is_file():
            try:
                saved_paths = json.loads(paths_path.read_text(encoding="utf-8"))
                cached_frame_embs = torch.load(emb_path, map_location="cpu").float()
                cached_paths_index = {p: i for i, p in enumerate(saved_paths)}
            except Exception as e:
                print(f"  [frame embed cache] load failed, recomputing: {e}")
                cached_frame_embs = None
                cached_paths_index = {}

        # ── Get the resident embedder (loads once, stays in GPU memory) ───────
        embedder = self._get_or_load_frame_embedder()

        # Always embed query fresh (1 forward pass — fast, text-only)
        with self._frame_embedder_inference_context():
            q_emb = _as_tensor(embedder.process([
                {"text": query, "instruction": "Retrieve frames relevant to the user's query."}
            ]))
        q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

        if cached_frame_embs is not None:
            # ── Fast path: look up pre-computed frame embeddings ────────────
            ordered_embs = []
            frames_to_embed = []
            frames_to_embed_idx = []
            for idx, item in enumerate(frame_items):
                cache_i = cached_paths_index.get(item["frame_path"])
                if cache_i is not None:
                    ordered_embs.append((idx, cached_frame_embs[cache_i].float()))
                else:
                    frames_to_embed.append(item)
                    frames_to_embed_idx.append(idx)

            # Embed only frames absent from cache (rare after pre-computation)
            if frames_to_embed:
                for i in range(0, len(frames_to_embed), batch_size):
                    batch = frames_to_embed[i : i + batch_size]
                    samples = [{"video": [f["frame_path"]]} for f in batch]
                    try:
                        with self._frame_embedder_inference_context():
                            embs = _as_tensor(embedder.process(samples))
                    except Exception:
                        embs = torch.zeros(len(batch), q_emb.shape[-1])
                    embs = embs / embs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    for item, emb in zip(batch, embs):
                        orig_idx = frames_to_embed_idx[frames_to_embed.index(item)]
                        ordered_embs.append((orig_idx, emb.float()))

            ordered_embs.sort(key=lambda x: x[0])
            frame_embs = torch.stack([e for _, e in ordered_embs], dim=0)
        else:
            # ── Slow path: embed all frames, then save cache ─────────────────
            all_embs = []
            for i in range(0, len(frame_items), batch_size):
                batch = frame_items[i : i + batch_size]
                samples = [{"video": [item["frame_path"]]} for item in batch]
                try:
                    with self._frame_embedder_inference_context():
                        embs = _as_tensor(embedder.process(samples))
                except Exception:
                    embs_list = []
                    for item in batch:
                        try:
                            with self._frame_embedder_inference_context():
                                e = _as_tensor(embedder.process([{"video": [item["frame_path"]]}]))
                        except Exception:
                            e = torch.zeros(1, q_emb.shape[-1])
                        embs_list.append(e)
                    embs = torch.cat(embs_list, dim=0)
                all_embs.append(embs)
            frame_embs = torch.cat(all_embs, dim=0).float()
            frame_embs_norm = frame_embs / frame_embs.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            try:
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(frame_embs_norm.cpu(), emb_path)
                paths_path.write_text(
                    json.dumps([item["frame_path"] for item in frame_items]),
                    encoding="utf-8",
                )
                print(f"  [frame embed cache] Saved {frame_embs_norm.shape[0]} embeddings → {emb_path}")
            except Exception as e:
                print(f"  [frame embed cache] save failed: {e}")
            frame_embs = frame_embs_norm

        # NOTE: embedder is intentionally NOT released here — stays resident
        sims = (q_emb @ frame_embs.T).squeeze(0)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)

        scored = [
            {
                "frame_path": item["frame_path"],
                "timestamp": item["timestamp"],
                "relevance_score": float(score),
            }
            for item, score in zip(frame_items, sims.tolist())
        ]

        scored.sort(key=lambda x: (-x["relevance_score"], x["timestamp"]))
        return scored[:top_k]

    def _rerank_frame_candidates(self, query: str, frames: list, top_k: int):
        def _fallback(items):
            return [
                {
                    "frame_path": item["frame_path"],
                    "timestamp": item["timestamp"],
                    "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                }
                for item in items[:top_k]
            ]

        if not query or not frames:
            return _fallback(frames)

        unique_frames = []
        seen = set()
        for item in frames:
            frame_path = str(item.get("frame_path") or "").strip()
            if not frame_path or not os.path.exists(frame_path):
                continue
            key = os.path.realpath(frame_path)
            if key in seen:
                continue
            seen.add(key)
            unique_frames.append(
                {
                    "frame_path": frame_path,
                    "timestamp": float(item.get("timestamp", 0.0) or 0.0),
                }
            )

        if not unique_frames:
            return []

        try:
            return self._qwen_score_frames(query, unique_frames, top_k)
        except Exception as e:
            print(f"  frame rerank qwen fallback: {e}")
            return _fallback(unique_frames)

    def _clip_path_to_segment(self, clip_path: str, score: float) -> dict | None:
        if not clip_path:
            return None

        base = os.path.basename(clip_path)
        stem, _ = os.path.splitext(base)
        parts = stem.split("_")

        # Expected format: clip_<idx>_<HH-MM-SS>_to_<HH-MM-SS>
        if len(parts) >= 5 and parts[0] == "clip" and parts[3] == "to":
            try:
                start_h, start_m, start_s = [int(x) for x in parts[2].split("-")]
                end_h, end_m, end_s = [int(x) for x in parts[4].split("-")]
                start = float(start_h * 3600 + start_m * 60 + start_s)
                end = float(end_h * 3600 + end_m * 60 + end_s)
                return {
                    "start": start,
                    "end": min(float(self.duration), end),
                    "confidence": float(score),
                }
            except Exception:
                pass

        try:
            clip_number = int(parts[1])
            start = float(clip_number * self.clip_duration)
            end = float(min(self.duration, start + self.clip_duration))
            return {
                "start": start,
                "end": end,
                "confidence": float(score),
            }
        except Exception:
            return None

    def _informative_clip_retrieval(self, query: str, top_k: int):
        if not query:
            return []

        result = self._run_qwen_temporal_grounder(query, top_k)
        segments = result.get("segments")
        if not isinstance(segments, list):
            return []
        return [
            {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "confidence": float(seg.get("confidence", 0.0) or 0.0),
            }
            for seg in segments
            if isinstance(seg, dict)
        ]

    def _frame_embedder_device_index(self) -> int:
        """GPU index reserved for the persistent frame embedder (GPU 1)."""
        raw = str(os.environ.get("FRAME_EMBEDDER_DEVICE_INDEX", "") or "").strip()
        if raw.isdigit():
            return int(raw)
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return 1 if count > 1 else 0

    def _temporal_grounder_device_index(self) -> int:
        """GPU index reserved for temporal grounder tools (kept off vLLM GPU 0 and frame-embedder GPU 1)."""
        requested = getattr(self, "temporal_grounder_device_index", None)
        if requested is not None:
            return int(requested)
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # ≥3 GPUs: 0=vLLM, 1=frame embedder, 2=temporal grounder
        if count >= 3:
            return 2
        return 1 if count > 1 else 0

    @contextmanager
    def _temporal_grounder_inference_context(self):
        tool_dev = self._temporal_grounder_device_index()
        use_bf16 = (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        with torch.cuda.device(tool_dev):
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    @contextmanager
    def _frame_embedder_inference_context(self):
        """Inference context for the resident frame embedder on its dedicated GPU (GPU 1)."""
        embed_dev = self._frame_embedder_device_index()
        use_bf16 = (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        with torch.cuda.device(embed_dev):
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def _temporal_grounder_model_kwargs(self):
        use_bf16 = (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        return {"torch_dtype": torch.bfloat16} if use_bf16 else {}

    def _release_temporal_grounder_runtime(self, runtime):
        if runtime is None:
            return
        tool_dev = self._temporal_grounder_device_index()
        model = getattr(runtime, "model", None)
        if isinstance(model, torch.nn.Module):
            try:
                model.cpu()
            except Exception:
                pass
        score_linear = getattr(runtime, "score_linear", None)
        if isinstance(score_linear, torch.nn.Module):
            try:
                score_linear.cpu()
            except Exception:
                pass
        for attr in ("model", "processor", "tokenizer", "score_linear"):
            if hasattr(runtime, attr):
                try:
                    setattr(runtime, attr, None)
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            with torch.cuda.device(tool_dev):
                torch.cuda.empty_cache()

    def _resolve_temporal_grounder_model_dir(self, model_name: str) -> str:
        raw = str(model_name or "").strip()
        if not raw:
            raise RuntimeError("temporal grounder model name is empty")

        path = Path(raw).expanduser()
        if path.exists():
            return str(path.resolve())

        cached = self._resolve_hf_snapshot(raw)
        if cached:
            return cached

        return snapshot_download(repo_id=raw)

    def _load_temporal_grounder_helper_class(
        self,
        model_dir: str,
        script_name: str,
        class_name: str,
        cache_attr: str,
        patch_sample_frames: bool = False,
    ):
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        module_path = Path(model_dir) / "scripts" / script_name
        if not module_path.exists():
            raise RuntimeError(f"Missing helper script: {module_path}")

        spec = importlib.util.spec_from_file_location(
            f"refiner_{class_name.lower()}_{abs(hash(str(module_path)))}",
            str(module_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load helper script: {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        klass = getattr(module, class_name)
        if patch_sample_frames and not hasattr(klass, "_sample_frames"):
            klass._sample_frames = staticmethod(module.sample_frames)
        setattr(self, cache_attr, klass)
        return klass

    def _temporal_grounder_video_info(self) -> dict:
        cached = getattr(self, "_temporal_grounder_video_info_cache", None)
        if isinstance(cached, dict) and cached.get("video_path") == self.video_path:
            return cached

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if video_fps <= 0:
            video_fps = max(1.0, float(getattr(self, "_video_fps", 24.0) or 24.0))
        if frame_count <= 0:
            frame_count = max(1, int(round(float(self.duration) * video_fps)))

        info = {
            "video_path": self.video_path,
            "video_fps": video_fps,
            "frame_count": frame_count,
            "duration": float(frame_count) / float(video_fps),
        }
        self._temporal_grounder_video_info_cache = info
        return info

    def _temporal_grounder_windows(self) -> list:
        info = self._temporal_grounder_video_info()
        duration = float(info["duration"])
        clip_seconds = max(0.1, float(self.clip_duration))
        stride_seconds = max(0.1, float(getattr(self, "temporal_grounder_stride_seconds", clip_seconds / 2.0)))

        starts = []
        t = 0.0
        while t < duration:
            starts.append(round(t, 3))
            t += stride_seconds
        last = max(0.0, duration - clip_seconds)
        if not starts or abs(starts[-1] - last) > 1e-3:
            starts.append(round(last, 3))

        windows = []
        for start in sorted(set(starts)):
            end = min(start + clip_seconds, duration)
            if end - start > 0.05:
                windows.append({"start": start, "end": end})
        return windows

    def _temporal_grounder_frames_per_window(self) -> int:
        sample_fps = max(0.1, float(getattr(self, "temporal_grounder_sample_fps", 1.0) or 1.0))
        max_frames = max(1, int(getattr(self, "temporal_grounder_max_frames", 64) or 64))
        return min(max_frames, max(1, int(round(float(self.clip_duration) * sample_fps))))

    def _temporal_grounder_sample_window_frames(
        self,
        start: float,
        end: float,
        num_frames: int,
    ) -> list:
        info = self._temporal_grounder_video_info()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        span = max(float(end) - float(start), 1e-3)
        times = [float(start) + (i + 0.5) * span / float(num_frames) for i in range(num_frames)]
        frames = []
        last_frame = None

        for t in times:
            frame_idx = max(
                0,
                min(
                    int(info["frame_count"]) - 1,
                    int(round(float(t) * float(info["video_fps"]))),
                ),
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                last_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if last_frame is not None:
                frames.append(last_frame.copy())

        cap.release()

        if not frames:
            raise RuntimeError(f"Could not decode frames for window {start:.2f}-{end:.2f}s")
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        return frames

    def _run_clip_temporal_grounder(self, query: str, top_k: int, warning: str | None = None) -> dict:
        try:
            result = self._run_qwen_temporal_grounder(query, top_k)
        except Exception as e:
            msg = str(e)
            print(f"  clip temporal grounder error: {msg}")
            result = {
                "query": query,
                "segments": [],
                "initial_segments": [],
                "reranked_segments": [],
                "video_duration": float(self.duration),
                "retrieval_backend": "qwen_embed_rerank",
                "warning": msg,
            }

        if warning:
            existing = str(result.get("warning", "") or "").strip()
            result["warning"] = f"{warning}; {existing}" if existing else warning
        return result

    def _run_qwen_temporal_grounder(self, query: str, top_k: int) -> dict:
        # NOTE: frame embedder eviction before this call is handled by
        # _execute_refine_plan which wraps temporal_grounder calls. Do NOT
        # release here — _execute_refine_plan restores it afterwards.

        def _as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float()
            return torch.tensor(x, dtype=torch.float32)

        model_dir = self._resolve_temporal_grounder_model_dir(
            getattr(self, "temporal_grounder_model_name", "Qwen/Qwen3-VL-Embedding-8B")
        )
        reranker_dir = self._resolve_temporal_grounder_model_dir(
            getattr(self, "temporal_grounder_reranker_model_name", "Qwen/Qwen3-VL-Reranker-8B")
        )
        Embedder = self._load_temporal_grounder_helper_class(
            model_dir,
            "qwen3_vl_embedding.py",
            "Qwen3VLEmbedder",
            "_temporal_grounder_embedder_class",
        )
        Reranker = self._load_temporal_grounder_helper_class(
            reranker_dir,
            "qwen3_vl_reranker.py",
            "Qwen3VLReranker",
            "_temporal_grounder_reranker_class",
            patch_sample_frames=True,
        )

        windows = self._temporal_grounder_windows()
        if not windows:
            return {
                "query": query,
                "segments": [],
                "initial_segments": [],
                "reranked_segments": [],
                "video_duration": float(self.duration),
                "retrieval_backend": "qwen_embed_rerank",
                "warning": "no temporal windows were created",
            }

        frames_per_window = self._temporal_grounder_frames_per_window()
        batch_size = max(1, int(getattr(self, "temporal_grounder_batch_size", 8) or 8))
        model_kwargs = self._temporal_grounder_model_kwargs()
        cache_meta = {
            "video_path": str(self.video_path),
            "model_dir": str(model_dir),
            "clip_seconds": float(self.clip_duration),
            "stride_seconds": float(getattr(self, "temporal_grounder_stride_seconds", max(1.0, float(self.clip_duration) / 2.0))),
            "sample_fps": float(getattr(self, "temporal_grounder_sample_fps", 1.0)),
            "frames_per_window": int(frames_per_window),
            "video_duration": float(self.duration),
            "dtype": "bfloat16" if bool(model_kwargs) else "float32",
        }

        clip_embeddings = None
        cached_windows, cached_embeddings = self._load_temporal_grounder_clip_embeddings_cache(cache_meta)
        if cached_windows is not None and cached_embeddings is not None:
            windows = cached_windows or windows
            clip_embeddings = cached_embeddings

        tool_dev_idx = self._temporal_grounder_device_index()
        with torch.cuda.device(tool_dev_idx):
            embedder = Embedder(
                model_name_or_path=str(model_dir),
                num_frames=frames_per_window,
                max_frames=frames_per_window,
                **model_kwargs,
            )
        if clip_embeddings is None:
            embs = []
            total_batches = (len(windows) + batch_size - 1) // batch_size
            for i in tqdm(
                range(0, len(windows), batch_size),
                total=total_batches,
                desc="Temporal grounder embedding",
                unit="batch",
            ):
                batch_windows = windows[i:i + batch_size]
                batch = [
                    {"video": self._temporal_grounder_sample_window_frames(seg["start"], seg["end"], frames_per_window)}
                    for seg in batch_windows
                ]
                with self._temporal_grounder_inference_context():
                    embs.append(_as_tensor(embedder.process(batch)))
            clip_embeddings = torch.cat(embs, dim=0).float()
            self._save_temporal_grounder_clip_embeddings_cache(cache_meta, windows, clip_embeddings)

        with self._temporal_grounder_inference_context():
            query_embedding = _as_tensor(embedder.process([{"text": query, "instruction": "Retrieve video clips relevant to the user's query."}]))

        scores = (query_embedding @ clip_embeddings.T).squeeze(0)
        top_k = min(int(top_k), len(windows))
        top_scores, top_indices = torch.topk(scores, k=top_k)
        initial_segments = [
            {
                "start": float(windows[idx]["start"]),
                "end": float(windows[idx]["end"]),
                "confidence": float(score),
                "embed_score": float(score),
            }
            for score, idx in zip(top_scores.tolist(), top_indices.tolist())
        ]

        self._release_temporal_grounder_runtime(embedder)
        embedder = None

        with torch.cuda.device(tool_dev_idx):
            reranker = Reranker(
                model_name_or_path=str(reranker_dir),
                num_frames=frames_per_window,
                max_frames=frames_per_window,
                **model_kwargs,
        )
        candidates = []
        for embed_score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            seg = dict(windows[idx])
            seg["embed_score"] = float(embed_score)
            seg["video"] = self._temporal_grounder_sample_window_frames(
                float(seg["start"]),
                float(seg["end"]),
                frames_per_window,
            )
            candidates.append(seg)

        with self._temporal_grounder_inference_context():
            rerank_scores = reranker.process(
                {
                    "instruction": "Retrieve video clips relevant to the user's query.",
                    "query": {"text": query},
                    "documents": [{"video": seg["video"]} for seg in candidates],
                }
            )

        self._release_temporal_grounder_runtime(reranker)
        reranker = None

        reranked_segments = []
        for seg, rerank_score in zip(candidates, rerank_scores):
            reranked_segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "confidence": float(rerank_score),
                    "embed_score": float(seg["embed_score"]),
                    "rerank_score": float(rerank_score),
                }
            )
        reranked_segments.sort(
            key=lambda x: (-float(x.get("confidence", 0.0) or 0.0), float(x.get("start", 0.0))),
        )

        print(f"  [Temporal Grounder] query: {query!r}")
        for rank, seg in enumerate(reranked_segments[:3], start=1):
            print(
                f"  [Temporal Grounder] top-{rank}: "
                f"{seg['start']:.1f}s–{seg['end']:.1f}s  "
                f"rerank={seg['rerank_score']:.4f}  embed={seg['embed_score']:.4f}"
            )

        return {
            "query": query,
            "segments": reranked_segments,
            "initial_segments": initial_segments,
            "reranked_segments": reranked_segments,
            "video_duration": float(self.duration),
            "retrieval_backend": "qwen_embed_rerank",
        }

    def _execute_refine_tool_call(self, tool_name: str, arguments: dict) -> str:
        handlers = {
            "temporal_grounder": self._process_temporal_grounder,
            "frame_retriever": self._process_frame_retriever,
            "asr": self._process_asr,
            "audio_grounder": self._process_audio_grounder,
            "ocr": self._process_ocr,
            "spatial_grounder": self._process_spatial_grounder,
            "counter": self._process_counter,
            "dense_captioner": self._process_dense_captioner,
            "action_recognizer": self._process_action_recognizer,
            "chart_analyzer": self._process_chart_analyzer,
            # "video_qa_reanswerer": self._process_video_qa_reanswerer,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}\n"

        payload = json.dumps(
            {"tool_calls": [{"tool": tool_name, "arguments": arguments or {}}]},
            ensure_ascii=False,
        )
        return handler(payload)

    def _execute_refine_plan(self, planner_plan: dict) -> list:
        if not isinstance(planner_plan, dict):
            return []

        print("\n" + "=" * 70)
        print("planner_plan: ", planner_plan)
        print("=" * 70 + "\n")

        tool_calls = planner_plan.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return []

        ordered_calls = sorted(
            [call for call in tool_calls if isinstance(call, dict)],
            key=lambda call: int(call.get("step", 0) or 0),
        )
        step_tools = {
            int(call.get("step", 0) or 0): str(call.get("tool", "") or "")
            for call in ordered_calls
            if int(call.get("step", 0) or 0)
        }

        print("\n" + "=" * 70)
        print("ordered_calls: ", ordered_calls)
        print("=" * 70 + "\n")

        execution_results = []
        step_results: dict = {}  # step_num (int) → parsed JSON output for dep resolution
        fallback_step_results = dict(getattr(self, "_refinement_prev_step_results", {}) or {})
        fallback_step_tools = dict(getattr(self, "_refinement_prev_step_tools", {}) or {})
        current_plan_steps = {
            int(call.get("step", 0) or 0)
            for call in ordered_calls
            if int(call.get("step", 0) or 0)
        }
        ibase = getattr(self, "_refinement_debug_iter_dir", None)

        for call in ordered_calls:
            step_num = int(call.get("step") or 0)
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            args_dict = arguments if isinstance(arguments, dict) else {}
            depends_on = [
                int(dep) for dep in (call.get("depends_on", []) or [])
                if str(dep).strip().isdigit()
            ]

            # Resolve any <STEPN:json.path> references from prior step outputs
            args_dict = self._resolve_step_refs(
                args_dict,
                step_results,
                fallback_step_results=fallback_step_results,
                blocked_steps=current_plan_steps - set(step_results.keys()),
            )
            args_dict = self._align_visual_tool_arguments(
                tool_name,
                args_dict,
                step_num,
                depends_on,
                step_results,
                step_tools,
                fallback_step_results=fallback_step_results,
                fallback_step_tools=fallback_step_tools,
            )

            tool_out_dir = None
            if ibase:
                slug = refiner_debug.sanitize_path_component(str(tool_name))
                tbase = Path(ibase) / f"tool_{step_num:02d}_{slug}"
                tool_out_dir = refiner_debug.ensure_outputs_dir(tbase)
                refiner_debug.write_json(tool_out_dir, "arguments.json", args_dict)
                self._refinement_debug_vlm_outputs_dir = tool_out_dir
                self._refinement_debug_vlm_input_basename = "model_input.json"

            try:
                args_dict = self._validate_tool_arguments(tool_name, args_dict)
            except Exception as e:
                output = self._format_refine_tool_result(
                    tool_name,
                    args_dict,
                    self._tool_validation_error_result(tool_name, e),
                )
                if tool_out_dir:
                    refiner_debug.write_text(tool_out_dir, "output.txt", (output or "").strip())
                execution_results.append(
                    {
                        "step": call.get("step"),
                        "tool": tool_name,
                        "arguments": args_dict,
                        "purpose": call.get("purpose", ""),
                        "depends_on": call.get("depends_on", []),
                        "output": (output or "").strip(),
                    }
                )
                continue

            try:
                output = self._execute_refine_tool_call(tool_name, args_dict)
            except Exception as tool_exc:
                print(f"  [Tool {step_num} {tool_name}] ERROR: {tool_exc}")
                output = f"Error executing {tool_name}: {tool_exc}"
            finally:
                self._refinement_debug_vlm_outputs_dir = None
                self._refinement_debug_vlm_input_basename = None

            if tool_out_dir:
                refiner_debug.write_text(tool_out_dir, "output.txt", (output or "").strip())

            # Store parsed output so later steps can reference it via <STEPN:...>
            if step_num:
                parsed = self._parse_tool_result_json(output or "")
                if isinstance(parsed, dict):
                    step_results[step_num] = parsed

            execution_results.append(
                {
                    "step": call.get("step"),
                    "tool": tool_name,
                    "arguments": args_dict,
                    "purpose": call.get("purpose", ""),
                    "depends_on": depends_on,
                    "output": (output or "").strip(),
                }
            )
        self._refinement_prev_step_results = dict(step_results)
        self._refinement_prev_step_tools = dict(step_tools)
        return execution_results

    def _get_asr_whisperx(self, start_time=None, end_time=None):
        global _WHISPERX_MODEL, _WHISPERX_MODEL_KEY, _WHISPERX_ALIGN, _WHISPERX_META
        if whisperx is None or torch is None:
            raise ImportError("whisperx or torch not installed")

        device_obj = self._whisperx_torch_device()
        device = str(device_obj)
        model_device, model_device_index = self._whisperx_ctranslate2_device(device_obj)
        compute_type = self._whisperx_compute_type(device_obj)
        model_config = self._whisperx_model_config()
        window = self._whisperx_effective_range(start_time, end_time)
        start = float(window["transcribe_start"])
        end = float(window["transcribe_end"])
        model_key = (
            str(model_config["model_spec"]),
            str(model_device),
            int(model_device_index),
            str(compute_type),
        )

        if _WHISPERX_MODEL is None or _WHISPERX_MODEL_KEY != model_key:
            with _whisperx_torch_load_compat():
                with self._whisperx_hub_access(
                    allow_download=not bool(model_config.get("local_files_only"))
                ):
                    _WHISPERX_MODEL = whisperx.load_model(
                        str(model_config["model_spec"]),
                        model_device,
                        device_index=model_device_index,
                        compute_type=compute_type,
                        download_root=model_config.get("download_root"),
                        local_files_only=bool(model_config.get("local_files_only")),
                    )
            _WHISPERX_MODEL_KEY = model_key
            _WHISPERX_ALIGN = None
            _WHISPERX_META = None
        audio = self._load_audio_with_ffmpeg(self.video_path)
        sample_rate = float(getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000))
        sample_start = max(0, int(start * sample_rate))
        sample_end = min(len(audio), int(end * sample_rate))
        audio_window = audio[sample_start:sample_end] if sample_end > sample_start else audio[0:0]
        if len(audio_window) == 0:
            return {
                "language_detected": "unknown",
                "transcript": "",
                "full_transcript": "",
                "segments": [],
                "words": [],
                "asr_backend": "whisperx",
            }

        transcribe_kwargs = {
            "batch_size": int(os.getenv("WHISPERX_BATCH", "8")),
        }
        language_hint = os.getenv("WHISPERX_LANGUAGE", "").strip()
        if language_hint:
            transcribe_kwargs["language"] = language_hint
        result = _WHISPERX_MODEL.transcribe(audio_window, **transcribe_kwargs)
        lang = result.get("language") or "en"
        segs = []
        try:
            if _WHISPERX_ALIGN is None or _WHISPERX_META is None:
                with _whisperx_torch_load_compat():
                    model_a, meta = whisperx.load_align_model(language_code=lang, device=device)
                _WHISPERX_ALIGN, _WHISPERX_META = model_a, meta
            aligned = whisperx.align(
                result["segments"],
                _WHISPERX_ALIGN,
                _WHISPERX_META,
                audio_window,
                device,
                return_char_alignments=False,
            )
            segs = aligned.get("segments", [])
        except Exception as ex:
            print(f"  WhisperX align skipped: {ex}")
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

        time_offset = float(start)
        normalized = []
        for s in segs:
            item = dict(s)
            item["start"] = float(item.get("start", 0)) + time_offset
            item["end"] = float(item.get("end", 0)) + time_offset
            words = []
            for w in item.get("words") or []:
                word_item = dict(w)
                if "start" in word_item:
                    word_item["start"] = float(word_item.get("start", 0)) + time_offset
                if "end" in word_item:
                    word_item["end"] = float(word_item.get("end", 0)) + time_offset
                words.append(word_item)
            item["words"] = words
            normalized.append(item)

        segments = []
        words_flat = []
        for s in normalized:
            segments.append(
                {
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", 0)),
                    "text": str(s.get("text", "")).strip(),
                    "speaker": s.get("speaker"),
                    "confidence": float(s.get("score", 0.9) or 0.9),
                }
            )
            for w in s.get("words") or []:
                words_flat.append(
                    {
                        "word": str(w.get("word", "")),
                        "start": float(w.get("start", 0)),
                        "end": float(w.get("end", 0)),
                        "confidence": float(w.get("score", 0.0) or 0.0),
                    }
                )
        transcript = " ".join(x["text"] for x in segments).strip()
        return {
            "language_detected": lang,
            "transcript": transcript,
            "full_transcript": transcript,
            "segments": segments,
            "words": words_flat,
            "asr_backend": "whisperx",
            "requested_range": {
                "start": float(window["requested_start"]),
                "end": float(window["requested_end"]),
            },
            "transcription_range": {
                "start": float(window["transcribe_start"]),
                "end": float(window["transcribe_end"]),
            },
            "window_expanded": bool(window["expanded"]),
        }

    def _process_temporal_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "temporal_grounder")
        if not calls:
            return ""

        print("\n[Tool] Temporal Grounder")
        results = []
        topk = int(getattr(self, "retrieval_top_k", 5))
        backend = str(getattr(self, "temporal_grounder_backend", "qwen") or "qwen").strip().lower()

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            result = None
            warning = None
            if query and backend in {"qwen", "qwen_embed_rerank"}:
                try:
                    result = self._run_qwen_temporal_grounder(query, topk)
                except Exception as e:
                    print(f"  qwen temporal grounder error: {e}")
                    warning = f"qwen temporal grounder error: {e}"

            if result is None:
                result = self._run_clip_temporal_grounder(query, topk, warning=warning)

            results.append(self._format_refine_tool_result("temporal_grounder", arguments, result))

        return "".join(results)

    def _process_frame_retriever(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "frame_retriever")
        if not calls:
            return ""

        print("\n[Tool] Frame Retriever")
        results = []

        for arguments in calls:
            query = str(arguments.get("query", "") or "").strip()
            timestamps = arguments.get("timestamps")
            num_frames = max(1, int(arguments.get("num_frames", 5) or 5))
            if isinstance(timestamps, str):
                timestamps = robust_eval(timestamps)

            # time_range may be injected by _align_visual_tool_arguments to restrict
            # the search to a temporal_grounder window even in the query-only path.
            time_range = arguments.get("time_range")  # (start_sec, end_sec) or None

            frames = []
            mode = "timestamp"

            if timestamps and query:
                candidates = []
                for ts in list(timestamps):
                    frame_path, frame_ts = self._get_frame_at_timestamp(ts)
                    if frame_path:
                        candidates.append(
                            {
                                "frame_path": frame_path,
                                "timestamp": float(frame_ts),
                            }
                        )
                frames = self._rerank_frame_candidates(query, candidates, min(3, num_frames))
                mode = "query"
            elif timestamps:
                for ts in list(timestamps)[:num_frames]:
                    frame_path, frame_ts = self._get_frame_at_timestamp(ts)
                    if frame_path:
                        frames.append(
                            {
                                "frame_path": frame_path,
                                "timestamp": float(frame_ts),
                            }
                        )
            elif query:
                mode = "query"
                try:
                    kind, matches = self._informative_retrieval(query, num_frames, time_range=time_range)
                except Exception as e:
                    print(f"  Error: {e}")
                    kind, matches = "dense", []

                if kind == "segment":
                    for seg in matches[:num_frames]:
                        start = float(seg.get("start", 0.0) or 0.0)
                        end = float(seg.get("end", start) or start)
                        score = float(seg.get("confidence", 0.0) or 0.0)
                        ts = (start + end) / 2.0
                        frame_path, frame_ts = self._get_frame_at_timestamp(ts)
                        if frame_path:
                            frames.append(
                                {
                                    "frame_path": frame_path,
                                    "timestamp": float(frame_ts),
                                    "relevance_score": float(score),
                                }
                            )
                else:
                    for frame_path, score in matches[:num_frames]:
                        ts = self._timestamp_from_dense_frame_path(frame_path)
                        frames.append(
                            {
                                "frame_path": frame_path,
                                "timestamp": float(ts),
                                "relevance_score": float(score),
                            }
                        )

            result = {"mode": mode, "frames": frames}
            results.append(self._format_refine_tool_result("frame_retriever", arguments, result))

        return "".join(results)

    def _process_asr(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "asr")
        if not calls:
            return ""

        print("\n[Tool] ASR")
        results = []

        for arguments in calls:
            result = None
            whisperx_error = None
            if os.getenv("REFINER_DISABLE_WHISPERX", "").strip() != "1":
                try:
                    if self._whisperx_sidecar_base_cmd():
                        result = self._get_asr_whisperx_sidecar(
                            arguments.get("start_time"),
                            arguments.get("end_time"),
                        )
                    else:
                        result = self._get_asr_whisperx(
                            arguments.get("start_time"),
                            arguments.get("end_time"),
                        )
                except Exception as e:
                    whisperx_error = f"{type(e).__name__}: {e}"
                    print(f"  WhisperX ASR fallback: {e}")
            else:
                whisperx_error = "WhisperX disabled by REFINER_DISABLE_WHISPERX=1"
            if result is None:
                result = self._get_asr_result_from_subtitles(
                    arguments.get("start_time"), arguments.get("end_time")
                )
                subtitle_has_content = bool(result.get("segments"))
                subtitle_source_available = bool(result.get("subtitle_source_available"))
                result["asr_backend"] = "subtitles" if subtitle_has_content else "none"
                result["asr_fallback_used"] = True
                result["asr_error"] = whisperx_error
                if subtitle_has_content:
                    result["asr_status"] = "subtitle_fallback"
                    result["note"] = "WhisperX failed; returning subtitle-derived transcript for the requested range."
                else:
                    result["asr_status"] = "unavailable"
                    if subtitle_source_available:
                        result["note"] = "WhisperX failed and subtitles exist, but the requested time range has no subtitle coverage."
                    else:
                        result["note"] = "WhisperX failed and no subtitle source is available for this video."
            else:
                result["asr_fallback_used"] = False
                result["asr_error"] = None
                result["asr_status"] = "ok"
            results.append(self._format_refine_tool_result("asr", arguments, result))

        return "".join(results)

    def _audio_grounder_clap(self, arguments: dict) -> dict:
        global _CLAP_MODULE, _CLAP_RUNTIME
        if np is None or torch is None or CLAP_Module is None:
            return {
                "query": str(arguments.get("query", "")).strip(),
                "query_mode": self._audio_grounder_query_mode(str(arguments.get("query", "")).strip()),
                "events": [],
                "audio_summary": "laion_clap or deps not available",
                "backend": "none",
            }

        query = str(arguments.get("query", "")).strip()
        query_mode = self._audio_grounder_query_mode(query)
        st = arguments.get("start_time")
        ed = arguments.get("end_time")
        t0, t1 = self._get_time_range(st, ed)
        try:
            if _CLAP_MODULE is None:
                clap_device = self._clap_torch_device()
                clap_ckpt = self._clap_checkpoint_path()
                if clap_ckpt is None and os.getenv("HF_HUB_OFFLINE", "").strip() == "1":
                    raise RuntimeError(
                        "LAION-CLAP checkpoint 630k-audioset-best.pt is not available in the local cache."
                    )
                clap_text_snapshot = self._clap_text_model_snapshot()
                if clap_text_snapshot is None and os.getenv("HF_HUB_OFFLINE", "").strip() == "1":
                    raise RuntimeError(self._clap_text_model_error())
                module = CLAP_Module(enable_fusion=False, device=str(clap_device))
                if clap_ckpt is not None:
                    module.load_ckpt(ckpt=str(clap_ckpt), verbose=False)
                else:
                    module.load_ckpt(verbose=False)
                _CLAP_MODULE = module
                _CLAP_RUNTIME = {
                    "device": str(clap_device),
                    "checkpoint": str(clap_ckpt) if clap_ckpt is not None else None,
                    "text_model_snapshot": clap_text_snapshot,
                }

            audio_data, sr, _, _ = self._load_audio_window_with_ffmpeg(t0, t1, sample_rate=48000)
            duration = len(audio_data) / float(sr)
            win = float(os.getenv("CLAP_WINDOW_SEC", "2.0"))
            hop = float(os.getenv("CLAP_HOP_SEC", "1.0"))
            thresh = float(os.getenv("CLAP_SIM_THRESHOLD", "0.25"))

            with torch.no_grad():
                t_emb = _CLAP_MODULE.get_text_embedding([query], use_tensor=False)
            t_emb = np.asarray(t_emb).reshape(-1).flatten()

            events = []
            t = 0.0
            while t + win <= duration + 1e-6:
                i0 = int(t * sr)
                i1 = int(min(len(audio_data), (t + win) * sr))
                if i1 <= i0:
                    break
                chunk = audio_data[i0:i1].astype(np.float32)
                chunk_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                try:
                    self._write_mono_wav(chunk_wav, chunk, int(sr))
                    with torch.no_grad():
                        a_emb = _CLAP_MODULE.get_audio_embedding_from_filelist(
                            [chunk_wav], use_tensor=False
                        )
                    a_emb = np.asarray(a_emb).reshape(-1).flatten()
                finally:
                    try:
                        os.unlink(chunk_wav)
                    except OSError:
                        pass
                sim = float(np.dot(a_emb, t_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(t_emb) + 1e-8))
                if sim >= thresh:
                    events.append(
                        {
                            "event_label": query,
                            "start": float(t0 + t),
                            "end": float(t0 + min(t + win, duration)),
                            "confidence": min(1.0, max(0.0, sim)),
                        }
                    )
                t += hop

            merged_events = self._merge_audio_grounder_events(events)
            return {
                "query": query,
                "query_mode": query_mode,
                "events": merged_events,
                "raw_event_count": len(events),
                "audio_summary": (
                    f"CLAP scan {len(events)} peaks; merged {len(merged_events)} events; query={query!r}"
                ),
                "backend": "laion_clap",
                "audio_status": "ok",
                "audio_error": None,
                "audio_fallback_used": False,
                "clap_device": _CLAP_RUNTIME.get("device"),
                "clap_checkpoint": _CLAP_RUNTIME.get("checkpoint"),
                "clap_text_model_snapshot": _CLAP_RUNTIME.get("text_model_snapshot"),
            }
        except Exception as e:
            if "ffmpeg" in type(e).__name__.lower():
                print(f"  ffmpeg audio extract failed: {e}")
                return {
                    "query": query,
                    "query_mode": query_mode,
                    "events": [],
                    "audio_summary": "audio extract failed",
                    "backend": "none",
                }
            raise

    def _process_audio_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "audio_grounder")
        if not calls:
            return ""

        print("\n[Tool] Audio Grounder")
        results = []

        for arguments in calls:
            result = None
            clap_error = None
            query = str(arguments.get("query", "")).strip()
            query_mode = self._audio_grounder_query_mode(query)

            if query_mode == "targeted" and os.getenv("REFINER_DISABLE_CLAP", "").strip() != "1":
                try:
                    result = self._audio_grounder_clap(arguments if isinstance(arguments, dict) else {})
                except Exception as e:
                    clap_error = f"{type(e).__name__}: {e}"
                    print(f"  LAION-CLAP fallback: {e}")
                if result is not None and result.get("backend") == "none" and clap_error is None:
                    clap_error = result.get("audio_summary")
            elif query_mode == "targeted":
                clap_error = "LAION-CLAP disabled by REFINER_DISABLE_CLAP=1"

            if result is None or not (result.get("events") or result.get("distinct_event_groups")):
                subtitle_result = self._audio_grounder_from_subtitle_tags(
                    arguments if isinstance(arguments, dict) else {}
                )
                if subtitle_result.get("events") or subtitle_result.get("distinct_event_groups"):
                    result = subtitle_result

            if result is None or not (result.get("events") or result.get("distinct_event_groups")):
                try:
                    allow_heuristic = query_mode != "targeted" or self._audio_grounder_allow_targeted_heuristic_fallback()
                    heuristic_result = (
                        self._audio_grounder_heuristic(arguments if isinstance(arguments, dict) else {})
                        if allow_heuristic
                        else None
                    )
                except Exception as e:
                    heuristic_result = None
                    print(f"  heuristic audio fallback: {e}")
                    if clap_error is None:
                        clap_error = f"{type(e).__name__}: {e}"
                if heuristic_result is not None:
                    result = heuristic_result

            if result is None:
                asr_result = self._get_asr_result_from_subtitles(
                    arguments.get("start_time"), arguments.get("end_time")
                )
                if query_mode == "targeted" and not self._audio_grounder_allow_targeted_heuristic_fallback():
                    summary = (
                        "Targeted non-speech audio grounding requires LAION-CLAP; heuristic fallback is disabled for targeted queries."
                    )
                else:
                    summary = (
                        "Speech subtitles are available in this range, but non-speech audio grounding could not be completed."
                        if asr_result["segments"]
                        else "Non-speech audio grounding could not be completed with the available backends."
                    )
                result = {
                    "query": query,
                    "query_mode": query_mode,
                    "events": [],
                    "audio_summary": summary,
                    "backend": "stub",
                    "audio_status": "unavailable",
                    "audio_error": clap_error,
                    "audio_fallback_used": True,
                }
            else:
                result.setdefault("query", query)
                result.setdefault("query_mode", query_mode)
                result.setdefault("audio_status", "ok")
                result.setdefault("audio_error", clap_error)
                result.setdefault("audio_fallback_used", result.get("backend") != "laion_clap")
            results.append(self._format_refine_tool_result("audio_grounder", arguments, result))

        return "".join(results)

    def _ocr_paddle(self, frame_path: str) -> list:
        global _PADDLE_OCR
        if PaddleOCR is None:
            raise ImportError("paddleocr not installed")

        if _PADDLE_OCR is None:
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en")
        raw = _PADDLE_OCR.ocr(frame_path, cls=True)
        detections = []
        if not raw or raw[0] is None:
            return detections
        for line in raw[0]:
            if not line or len(line) < 2:
                continue
            box, txt_conf = line[0], line[1]
            text, conf = (txt_conf[0], float(txt_conf[1])) if isinstance(txt_conf, (list, tuple)) else ("", 0.0)
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            detections.append(
                {
                    "text": str(text).strip(),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "text_type": "scene_text",
                }
            )
        return detections

    def _ocr_pytesseract(self, frame_path: str) -> list:
        if pytesseract is None:
            raise ImportError("pytesseract not installed")

        data = pytesseract.image_to_data(Image.open(frame_path), output_type=pytesseract.Output.DICT)
        detections = []
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            conf = self._safe_float(data["conf"][i], -1.0)
            if not text or conf < 0:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            detections.append(
                {
                    "text": text,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": conf / 100.0 if conf > 1 else conf,
                    "text_type": "scene_text",
                }
            )
        return detections

    def _process_ocr(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "ocr")
        if not calls:
            return ""

        print("\n[Tool] OCR")
        results = []

        for arguments in calls:
            frame_paths, frame_timestamps = self._resolve_frame_bundle(arguments)
            source = (
                list(frame_paths)
                if len(frame_paths) > 1
                else (frame_paths[0] if frame_paths else f"{self.video_path}@0.0")
            )
            result = {"source": source, "detections": [], "full_text": "", "ocr_backend": "none"}

            if frame_paths:
                detections = []
                if len(frame_paths) == 1:
                    frame_path = frame_paths[0]
                    if os.getenv("REFINER_DISABLE_PADDLEOCR", "").strip() != "1":
                        try:
                            detections = self._ocr_paddle(frame_path)
                            result["ocr_backend"] = "paddleocr"
                        except Exception as e:
                            print(f"  PaddleOCR fallback: {e}")
                    if not detections:
                        try:
                            detections = self._ocr_pytesseract(frame_path)
                            result["ocr_backend"] = "pytesseract"
                        except Exception:
                            detections = []
                if detections:
                    result = {
                        "source": source,
                        "detections": detections,
                        "full_text": "\n".join(x["text"] for x in detections),
                        "ocr_backend": result.get("ocr_backend", "pytesseract"),
                    }
                else:
                    prompt = (
                        ocr_prompt.strip()
                        + (
                            "\n\nExtract all visible text across all provided frames together. Return JSON only.\n"
                            if len(frame_paths) > 1
                            else "\n\nExtract all visible text from this frame. Return JSON only.\n"
                        )
                    )
                    if len(frame_paths) > 1 and not self._use_vlm_remote_api():
                        merged_detections = []
                        merged_text = []
                        merged_raw = []
                        for frame_path, frame_ts in zip(frame_paths, frame_timestamps):
                            single_default = {
                                "source": frame_path,
                                "detections": [],
                                "full_text": "",
                                "ocr_backend": "none",
                            }
                            single_result = self._run_vlm_json(
                                prompt,
                                [frame_path],
                                [frame_ts],
                                single_default,
                            )
                            if not isinstance(single_result, dict):
                                continue
                            for det in (single_result.get("detections") or []):
                                if isinstance(det, dict):
                                    merged_detections.append(det)
                            text = str(single_result.get("full_text", "") or "").strip()
                            if text:
                                merged_text.append(text)
                            raw = str(single_result.get("raw_output", "") or "").strip()
                            if raw:
                                merged_raw.append(raw)
                        result = {
                            "source": source,
                            "detections": merged_detections,
                            "full_text": "\n".join(merged_text),
                            "ocr_backend": "vlm",
                        }
                        if merged_raw and not merged_detections and not merged_text:
                            result["raw_output"] = "\n\n".join(merged_raw)
                    else:
                        result = self._run_vlm_json(
                            prompt,
                            frame_paths,
                            frame_timestamps,
                            result,
                        )
                        if isinstance(result, dict):
                            result.setdefault("source", source)
                            result["ocr_backend"] = "vlm"

            results.append(self._format_refine_tool_result("ocr", arguments, result))

        return "".join(results)

    def _process_spatial_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "spatial_grounder")
        if not calls:
            return ""

        print("\n[Tool] Spatial Grounder")
        results = []

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            frame_paths, frame_timestamps = self._resolve_frame_bundle(arguments)
            return_masks = bool(arguments.get("return_masks", False))

            if not frame_paths:
                frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)
                frame_path = arguments.get("frame_path")
                if not frame_path or not os.path.exists(frame_path):
                    frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)
                if frame_path and os.path.exists(frame_path):
                    frame_paths = [frame_path]
                    frame_timestamps = [float(frame_ts or 0.0)]

            backend = str(getattr(self, "spatial_grounder_backend", "grounding_dino") or "grounding_dino").strip().lower()
            frame_results = []
            for frame_path, frame_ts in zip(frame_paths, frame_timestamps):
                if backend == "grounding_dino":
                    result, warning = self._run_grounding_dino_spatial_grounder(
                        frame_path,
                        query,
                        return_masks=return_masks,
                    )
                    if result is None and bool(getattr(self, "spatial_grounder_vlm_fallback", True)):
                        result = self._run_spatial_grounder_vlm(
                            query,
                            frame_path,
                            frame_ts,
                            return_masks=return_masks,
                        )
                        if isinstance(result, dict):
                            result["backend"] = "vlm_fallback"
                            if warning:
                                result["warning"] = warning
                    elif result is None:
                        result = {
                            "query": query,
                            "detections": [],
                            "spatial_description": "",
                            "backend": "grounding_dino",
                            "warning": warning or "Grounding DINO inference failed.",
                        }
                else:
                    result = self._run_spatial_grounder_vlm(
                        query,
                        frame_path,
                        frame_ts,
                        return_masks=return_masks,
                    )

                if not isinstance(result, dict):
                    result = {
                        "query": query,
                        "detections": [],
                        "spatial_description": "",
                        "backend": backend,
                    }

                frame_results.append(
                    {
                        "frame_path": frame_path,
                        "timestamp": float(frame_ts),
                        **result,
                    }
                )

            if len(frame_results) <= 1:
                result = frame_results[0] if frame_results else {
                    "query": query,
                    "detections": [],
                    "spatial_description": "",
                    "backend": backend,
                }
            else:
                def _score(item):
                    dets = item.get("detections") or []
                    if isinstance(dets, list) and dets:
                        return max(float(det.get("confidence", 0.0) or 0.0) for det in dets if isinstance(det, dict))
                    return 0.0

                ranked = sorted(
                    frame_results,
                    key=lambda item: (-_score(item), -len(item.get("detections") or []), float(item.get("timestamp", 0.0) or 0.0)),
                )
                best = ranked[0]
                result = {
                    "query": query,
                    "frames": frame_results,
                    "best_frame": best,
                    "detections": list(best.get("detections") or []),
                    "spatial_description": str(best.get("spatial_description", "") or ""),
                    "backend": "multi_frame",
                    "selection_reason": "Top-level fields mirror one high-salience candidate frame; inspect frames[] for all frame-level grounding results.",
                }
            results.append(self._format_refine_tool_result("spatial_grounder", arguments, result))

        return "".join(results)

    def _process_counter(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "counter")
        if not calls:
            return ""

        print("\n[Tool] Counter")
        results = []

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            frame_paths, frame_timestamps = self._resolve_frame_bundle(arguments)
            if not frame_paths:
                frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)
                frame_path = arguments.get("frame_path")
                if not frame_path or not os.path.exists(frame_path):
                    frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)
                if frame_path and os.path.exists(frame_path):
                    frame_paths = [frame_path]
                    frame_timestamps = [float(frame_ts or 0.0)]

            default_result = {
                "query": query,
                "count": 0,
                "confidence": 0.0,
                "detections": [],
                "notes": "",
            }
            prompt = (
                counter_prompt.strip()
                + f"\n\nQuery: {query}\nReturn JSON only matching the OUTPUT FORMAT above.\n"
            )

            frame_results = []
            for frame_path, frame_ts in zip(frame_paths, frame_timestamps):
                single_result = self._run_vlm_json(
                    prompt,
                    [frame_path] if frame_path else [],
                    [float(frame_ts)],
                    dict(default_result),
                )
                if not isinstance(single_result, dict):
                    single_result = dict(default_result)
                frame_results.append(
                    {
                        "frame_path": frame_path,
                        "timestamp": float(frame_ts),
                        **single_result,
                    }
                )

            if len(frame_results) <= 1:
                result = frame_results[0] if frame_results else dict(default_result)
            else:
                ranked = sorted(
                    frame_results,
                    key=lambda item: (-float(item.get("confidence", 0.0) or 0.0), -int(item.get("count", 0) or 0), float(item.get("timestamp", 0.0) or 0.0)),
                )
                best = ranked[0]
                result = {
                    "query": query,
                    "frames": frame_results,
                    "best_frame": best,
                    "count": int(best.get("count", 0) or 0),
                    "confidence": float(best.get("confidence", 0.0) or 0.0),
                    "detections": list(best.get("detections") or []),
                    "notes": str(best.get("notes", "") or ""),
                    "selection_reason": "Top-level fields mirror one high-salience candidate frame; inspect frames[] for all frame-level counting results.",
                }
            results.append(self._format_refine_tool_result("counter", arguments, result))

        return "".join(results)

    def _run_dense_captioner_interval(
        self,
        start_time=None,
        end_time=None,
        granularity: str = "segment",
        focus_query: str = "",
    ) -> dict:
        """Run dense_captioner on [start_time, end_time]; return parsed JSON dict (same as tool output)."""
        granularity = str(granularity or "segment")
        fps = 1.0 if granularity == "frame" else 2.0
        frame_paths, timestamps, start, end = self._get_frames_for_range(
            start_time, end_time, fps=fps
        )
        focus_query = str(focus_query or "").strip()
        default_result = {
            "video_duration": float(self.duration),
            "captioned_range": {"start": start, "end": end},
            "captions": [],
            "overall_summary": "",
        }
        prompt = (
            dense_captioner_prompt.strip()
            + (
                f"\n\nRequested segment: {start:.3f}s to {end:.3f}s."
                " The attached frames are in chronological order from this interval."
                " Use absolute seconds within this requested segment for"
                " `captioned_range.start`, `captioned_range.end`, and every"
                " `captions[].start` / `captions[].end` field."
            )
            + f"\nGranularity: {granularity}. Focus query: {focus_query}\nReturn JSON only.\n"
        )
        return self._run_vlm_json(
            prompt,
            frame_paths,
            timestamps,
            default_result,
            force_local=True,
        )

    def _process_dense_captioner(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "dense_captioner")
        if not calls:
            return ""

        print("\n[Tool] Dense Captioner")
        results = []

        for arguments in calls:
            granularity = str(arguments.get("granularity", "segment") or "segment")
            focus_query = str(arguments.get("focus_query", "")).strip()
            result = self._run_dense_captioner_interval(
                arguments.get("start_time"),
                arguments.get("end_time"),
                granularity=granularity,
                focus_query=focus_query,
            )
            results.append(self._format_refine_tool_result("dense_captioner", arguments, result))

        return "".join(results)

    def _process_action_recognizer(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "action_recognizer")
        if not calls:
            return ""

        print("\n[Tool] Action Recognizer")
        results = []

        for arguments in calls:
            frame_paths, timestamps, start, end = self._get_frames_for_range(
                arguments.get("start_time"), arguments.get("end_time"), fps=None
            )
            query = str(arguments.get("query", "")).strip()
            default_result = {
                "analyzed_range": {"start": start, "end": end},
                "actions": [],
                "query_response": None,
            }
            prompt = (
                action_recognizer_prompt.strip()
                + f"\n\nFocus on human actions. Query: {query}\nReturn JSON only.\n"
            )
            result = self._run_vlm_json(prompt, frame_paths, timestamps, default_result)
            results.append(self._format_refine_tool_result("action_recognizer", arguments, result))

        return "".join(results)

    def _process_chart_analyzer(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "chart_analyzer")
        if not calls:
            return ""

        print("\n[Tool] Chart Analyzer")
        results = []

        for arguments in calls:
            frame_paths, frame_timestamps = self._resolve_frame_bundle(arguments)
            query = str(arguments.get("query", "") or "").strip()

            default_result = {
                "chart_type": "unknown",
                "title": "",
                "axes": {},
                "series": [],
                "key_observations": [],
                "relationships": [],
                "query_response": None,
            }

            if not frame_paths:
                default_result["query_response"] = "chart_analyzer unavailable or frame not found"
                results.append(self._format_refine_tool_result("chart_analyzer", arguments, default_result))
                continue

            prompt_text = chart_analyzer_prompt.strip()
            if query:
                prompt_text += f"\n\nQuery: {query}\nReturn JSON only matching the OUTPUT FORMAT above.\n"
            else:
                prompt_text += "\n\nReturn JSON only matching the OUTPUT FORMAT above.\n"

            def _run_chart_analysis(local_frame_paths, local_timestamps):
                mode = getattr(self, "chart_mode", "api")
                raw_output = None
                parsed = None
                merged = None
                try:
                    if mode == "api":
                        raw_output = self._call_chart_vision_api(prompt_text, local_frame_paths)
                        print(f'chart analyzer output: {raw_output}')
                        parsed = self._extract_json_payload(raw_output)
                    elif mode == "vlm":
                        merged = self._run_vlm_json(
                            prompt_text,
                            local_frame_paths,
                            local_timestamps,
                            dict(default_result),
                            force_local=True,
                        )
                        if isinstance(merged, dict) and "raw_output" in merged:
                            parsed = self._extract_json_payload(merged.get("raw_output", ""))
                        elif isinstance(merged, dict):
                            parsed = merged
                    elif mode == "internvl":
                        self._load_chart_model()
                        pixel_values = self._internvl_load_image(local_frame_paths[0])
                        generation_config = dict(max_new_tokens=1024, do_sample=False)
                        raw_output = self._chart_model.chat(
                            self._chart_tokenizer,
                            pixel_values,
                            prompt_text,
                            generation_config,
                        )
                        parsed = self._extract_json_payload(raw_output)
                    else:
                        fallback = dict(default_result)
                        fallback["query_response"] = f"unknown chart_mode: {mode}"
                        return fallback

                    if isinstance(parsed, dict):
                        result = parsed
                        result.setdefault("query_response", None)
                        return result

                    fallback_text = ""
                    if isinstance(merged, dict):
                        fallback_text = str(merged.get("raw_output", "") or "").strip()
                    if not fallback_text:
                        fallback_text = (raw_output or "").strip() if raw_output else ""
                    fallback = dict(default_result)
                    fallback["query_response"] = fallback_text
                    return fallback
                except Exception as e:
                    print(f"  Chart analyzer error: {e}")
                    fallback = dict(default_result)
                    fallback["query_response"] = f"chart_analyzer error: {e}"
                    return fallback

            if len(frame_paths) > 1:
                frame_results = []
                for frame_path, frame_timestamp in zip(frame_paths, frame_timestamps):
                    single_result = _run_chart_analysis([frame_path], [frame_timestamp])
                    frame_results.append(
                        {
                            "frame_path": frame_path,
                            "timestamp": float(frame_timestamp),
                            "score": float(self._score_chart_analysis_result(single_result)),
                            "result": single_result,
                        }
                    )
                result = self._select_chart_analysis_frame_result(frame_results)
            else:
                result = _run_chart_analysis(frame_paths, frame_timestamps)

            results.append(self._format_refine_tool_result("chart_analyzer", arguments, result))

        return "".join(results)

    def _process_math_solver(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "math_solver")
        if not calls:
            return ""

        print("\n[Tool] Math Solver")
        results = []

        for arguments in calls:
            question = str(arguments.get("question", "") or getattr(self, "question", "") or "").strip()
            answer_choices = arguments.get("answer_choices")
            if isinstance(answer_choices, list):
                choices = [str(item).strip() for item in answer_choices if str(item).strip()]
            else:
                choices = [str(item).strip() for item in (getattr(self, "options", None) or []) if str(item).strip()]
            question = self._strip_math_solver_answer_choice_text(question, choices)

            evidence = arguments.get("evidence", "")
            if isinstance(evidence, list):
                evidence_items = [str(item).strip() for item in evidence if str(item).strip()]
            else:
                text = str(evidence or "").strip()
                evidence_items = [text] if text else []
            evidence_items = [
                cleaned
                for cleaned in (
                    self._strip_math_solver_answer_choice_text(item, choices)
                    for item in evidence_items
                )
                if cleaned
            ]
            evidence_items = self._sanitize_math_solver_evidence_items(evidence_items)
            frame_paths, frame_timestamps = self._resolve_frame_bundle(arguments)

            default_result = {
                "interpreted_facts": [],
                "derivation": [],
                "result": None,
                "answer_choice": None,
                "confidence": 0.0,
                "insufficient_information": False,
                "missing_facts": [],
            }

            evidence_block = "\n".join(
                f"- {item}" for item in evidence_items
            ) or "- (no grounded evidence provided)"
            prompt = (
                math_solver_prompt.strip()
                + "\n\nQUESTION:\n"
                + (question or "(missing)")
                + "\n\nGROUNDED_EVIDENCE:\n"
                + evidence_block
            )

            if frame_paths:
                frame_block = "\n".join(
                    f"- {float(ts):.3f}s: {os.path.basename(path)}"
                    for path, ts in zip(frame_paths, frame_timestamps)
                )
                prompt += (
                    "\n\nSUPPORTING_FRAMES:\n"
                    + frame_block
                    + "\nUse the attached frame(s) as supporting visual context for directly visible"
                    + " labels, shapes, geometry relations, chart structure, and other plainly"
                    + " visible premises. Prefer directly visible primitive facts from the attached"
                    + " frame(s) for labels, shape type, attachment points, and intersections. If"
                    + " a textual claim and the attached frame suggest different directly visible"
                    + " primitives, treat that primitive as ambiguous rather than blindly preferring"
                    + " either source. Never invent facts that are unclear or hidden."
                )

            raw_output = ""
            if frame_paths:
                raw_output = self._call_math_solver_with_frames(prompt, frame_paths, frame_timestamps)
            if not raw_output:
                raw_output = self._call_math_solver_text(prompt)
            parsed = self._extract_json_payload(raw_output)

            if isinstance(parsed, dict):
                result = dict(default_result)
                result.update(parsed)
            else:
                result = dict(default_result)
                result["insufficient_information"] = True
                result["missing_facts"] = ["math_solver returned malformed or non-JSON output"]
                result["raw_output"] = str(raw_output or "").strip()

            if not isinstance(result.get("interpreted_facts"), list):
                result["interpreted_facts"] = [str(result.get("interpreted_facts", "")).strip()] if str(result.get("interpreted_facts", "")).strip() else []
            else:
                result["interpreted_facts"] = [str(item).strip() for item in result.get("interpreted_facts") if str(item).strip()]

            if not isinstance(result.get("derivation"), list):
                result["derivation"] = [str(result.get("derivation", "")).strip()] if str(result.get("derivation", "")).strip() else []
            else:
                result["derivation"] = [str(item).strip() for item in result.get("derivation") if str(item).strip()]

            if not isinstance(result.get("missing_facts"), list):
                result["missing_facts"] = [str(result.get("missing_facts", "")).strip()] if str(result.get("missing_facts", "")).strip() else []
            else:
                result["missing_facts"] = [str(item).strip() for item in result.get("missing_facts") if str(item).strip()]

            try:
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0) or 0.0)))
            except Exception:
                result["confidence"] = 0.0

            raw_insufficient = result.get("insufficient_information", False)
            if isinstance(raw_insufficient, str):
                result["insufficient_information"] = raw_insufficient.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                result["insufficient_information"] = bool(raw_insufficient)
            if result.get("result") is not None:
                result["result"] = str(result.get("result")).strip() or None
            if result.get("answer_choice") is not None:
                result["answer_choice"] = str(result.get("answer_choice")).strip() or None

            result = self._enforce_math_solver_consistency(evidence, result)
            result["answer_choice"] = self._match_math_solver_answer_choice(
                result.get("result"),
                choices,
                hinted_choice=result.get("answer_choice"),
                insufficient_information=bool(result.get("insufficient_information")),
            )
            results.append(self._format_refine_tool_result("math_solver", arguments, result))

        return "".join(results)

    def _internvl_load_image(self, frame_path: str):
        """Preprocess a single image for InternVL2 inference."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        input_size = 448

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        device = self._chart_torch_device()
        img = Image.open(frame_path).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16).to(device)
        return pixel_values

    # def _process_video_qa_reanswerer(self, output_text: str) -> str:
    #     calls = self._get_refine_tool_calls(output_text, "video_qa_reanswerer")
    #     if not calls:
    #         return ""
    #
    #     print("\n[Tool] Video QA Re-answerer")
    #     results = []
    #
    #     for arguments in calls:
    #         question = str(arguments.get("question", "")).strip()
    #         frame_paths, timestamps, _, _ = self._get_frames_for_range(None, None, fps=None)
    #         default_result = {
    #             "question": question,
    #             "answer": "",
    #             "reasoning": "",
    #             "confidence": 0.0,
    #             "key_evidence": [],
    #         }
    #         prompt = (
    #             video_qa_reanswerer_prompt.strip()
    #             + f"\n\nQuestion: {question}\nReturn JSON only.\n"
    #         )
    #         result = self._run_vlm_json(prompt, frame_paths, timestamps, default_result)
    #         results.append(self._format_refine_tool_result("video_qa_reanswerer", arguments, result))
    #
    #     return "".join(results)
