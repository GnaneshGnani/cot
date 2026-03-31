import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import numpy as np
import torch
import torchvision.transforms as T
import whisperx
import soundfile as sf
from laion_clap import CLAP_Module
from openai import OpenAI
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import refiner_debug
from refiner_utils import openai_chat_completion_limit_kwargs, openai_chat_temperature_kwargs
from refine_prompt import (
    action_recognizer_prompt,
    chart_analyzer_prompt,
    counter_prompt,
    dense_captioner_prompt,
    ocr_prompt,
    spatial_grunder_prompt,
    # video_qa_reanswerer_prompt,
)
from video_utils import robust_eval

_WHISPERX_MODEL = None
_WHISPERX_ALIGN = None
_WHISPERX_META = None
_PADDLE_OCR = None
_CLAP_MODULE = None
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

        if not isinstance(text, str):
            return None
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            if not (candidate.startswith("{") or candidate.startswith("[")):
                continue
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

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
        env = dict(os.environ)
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
        device_obj = self._whisperx_torch_device()
        compute_type = self._whisperx_compute_type(device_obj)
        model_name = os.getenv("WHISPERX_MODEL", "small")
        sidecar_script = os.path.join(_eval_dir, "whisperx_sidecar.py")
        cmd = base_cmd + [
            sidecar_script,
            "--video-path",
            self.video_path,
            "--start-time",
            str(window["transcribe_start"]),
            "--end-time",
            str(window["transcribe_end"]),
            "--model-name",
            model_name,
            "--device",
            str(device_obj),
            "--aux-device",
            self._whisperx_aux_device(),
            "--compute-type",
            compute_type,
            "--batch-size",
            str(int(os.getenv("WHISPERX_BATCH", "8"))),
        ]
        language_hint = os.getenv("WHISPERX_LANGUAGE", "").strip()
        if language_hint:
            cmd.extend(["--language", language_hint])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._whisperx_sidecar_env(base_cmd),
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
                        frame_ts = float(self.retriever._timestamp_from_dense_frame_path(frame_path))
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

        for seg in ranked_segments:
            aligned = [
                frame for frame in normalized_frames
                if seg["start"] - tol <= frame["timestamp"] <= seg["end"] + tol
            ]
            if aligned:
                return sorted(
                    aligned,
                    key=lambda frame: (
                        -frame["relevance_score"],
                        abs(frame["timestamp"] - seg["mid"]),
                        frame["timestamp"],
                    ),
                )

        best_seg = ranked_segments[0]
        nearest = sorted(
            normalized_frames,
            key=lambda frame: (
                abs(frame["timestamp"] - best_seg["mid"]),
                -frame["relevance_score"],
                frame["timestamp"],
            ),
        )
        return nearest[:1]

    def _align_visual_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
        current_step: int,
        depends_on: list,
        step_results: dict,
        step_tools: dict,
    ) -> dict:
        if tool_name not in {"spatial_grounder", "counter", "chart_analyzer", "ocr"}:
            return arguments

        frame_step = None
        for dep in depends_on:
            if step_tools.get(dep) == "frame_retriever" and isinstance(step_results.get(dep), dict):
                frame_step = dep
        if frame_step is None:
            return arguments

        temporal_step = None
        for dep in depends_on:
            if step_tools.get(dep) == "temporal_grounder" and isinstance(step_results.get(dep), dict):
                temporal_step = dep
        if temporal_step is None:
            temporal_step = self._latest_prior_step_of_type(frame_step, step_tools, "temporal_grounder")
        if temporal_step is None:
            temporal_step = self._latest_prior_step_of_type(current_step, step_tools, "temporal_grounder")
        if temporal_step is None:
            return arguments

        aligned_frames = self._select_frames_aligned_with_temporal_grounder(
            step_results.get(frame_step) or {},
            step_results.get(temporal_step) or {},
        )
        if not aligned_frames:
            return arguments

        updated = dict(arguments)
        if tool_name in {"chart_analyzer", "ocr"}:
            updated["frame_path"] = [
                {"frame_path": item["frame_path"], "timestamp": item["timestamp"]}
                for item in aligned_frames
            ]
            updated["timestamp"] = None
        else:
            updated["frame_path"] = aligned_frames[0]["frame_path"]
            updated["timestamp"] = aligned_frames[0]["timestamp"]
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

    def _call_chart_vision_api(self, prompt_text: str, frame_path: str) -> str:
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
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        vlm_out = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
        if vlm_out:
            refiner_debug.write_json(
                vlm_out,
                "model_input.json",
                {"model": self.chart_model_name, "messages": messages, "frame_path": frame_path},
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

    def _informative_retrieval(self, query: str, top_k: int):
        """Dense-frame text–image retrieval by default; optional clip-level search."""
        if getattr(self, "use_clip_retrieval", False):
            return "clip", self.retriever.get_informative_clips(
                query,
                video_path=self.video_path,
                top_k=top_k,
                total_duration=self.duration,
            )
        return "dense", self.retriever.get_informative_dense_frames(
            query,
            self.video_path,
            self.dataset_folder,
            
            top_k=top_k,
            total_duration=float(self.duration),
            dense_sample_fps=float(getattr(self, "dense_frame_fps", 24.0)),
            embed_batch=int(getattr(self, "dense_frame_embed_batch", 8)),
        )

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
            args_dict = self._resolve_step_refs(args_dict, step_results)
            args_dict = self._align_visual_tool_arguments(
                tool_name,
                args_dict,
                step_num,
                depends_on,
                step_results,
                step_tools,
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
        return execution_results

    def _get_asr_whisperx(self, start_time=None, end_time=None):
        global _WHISPERX_MODEL, _WHISPERX_ALIGN, _WHISPERX_META
        if whisperx is None or torch is None:
            raise ImportError("whisperx or torch not installed")

        device_obj = self._whisperx_torch_device()
        device = str(device_obj)
        model_device, model_device_index = self._whisperx_ctranslate2_device(device_obj)
        compute_type = self._whisperx_compute_type(device_obj)
        model_name = os.getenv("WHISPERX_MODEL", "small")
        window = self._whisperx_effective_range(start_time, end_time)
        start = float(window["transcribe_start"])
        end = float(window["transcribe_end"])

        if _WHISPERX_MODEL is None:
            with _whisperx_torch_load_compat():
                _WHISPERX_MODEL = whisperx.load_model(
                    model_name,
                    model_device,
                    device_index=model_device_index,
                    compute_type=compute_type,
                )
        audio = whisperx.load_audio(self.video_path)
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

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            segments = []
            if query:
                try:
                    kind, matches = self._informative_retrieval(query, topk)
                    if kind == "clip":
                        for clip_path, score in matches:
                            clip_number = int(os.path.basename(clip_path).split("_")[1])
                            start = float(clip_number * self.clip_duration)
                            end = float(min(self.duration, start + self.clip_duration))
                            segments.append(
                                {
                                    "start": start,
                                    "end": end,
                                    "confidence": float(score),
                                }
                            )
                    else:
                        half = float(getattr(self, "dense_segment_half_width", 0.5))
                        for frame_path, score in matches:
                            t = self.retriever._timestamp_from_dense_frame_path(frame_path)
                            segments.append(
                                {
                                    "start": max(0.0, t - half),
                                    "end": min(float(self.duration), t + half),
                                    "confidence": float(score),
                                }
                            )
                except Exception as e:
                    print(f"  Error: {e}")

            result = {
                "query": query,
                "segments": sorted(segments, key=lambda x: x["start"]),
                "video_duration": float(self.duration),
            }
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

            frames = []
            mode = "timestamp"

            if timestamps:
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
                    kind, matches = self._informative_retrieval(query, num_frames)
                except Exception as e:
                    print(f"  Error: {e}")
                    kind, matches = "dense", []

                if kind == "clip":
                    for clip_path, score in matches[:num_frames]:
                        clip_number = int(os.path.basename(clip_path).split("_")[1])
                        ts = clip_number * self.clip_duration + self.clip_duration / 2
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
                        ts = self.retriever._timestamp_from_dense_frame_path(frame_path)
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
        global _CLAP_MODULE
        if np is None or sf is None or torch is None or CLAP_Module is None:
            return {
                "query": str(arguments.get("query", "")).strip(),
                "events": [],
                "audio_summary": "laion_clap or deps not available",
                "backend": "none",
            }

        query = str(arguments.get("query", "")).strip()
        st = arguments.get("start_time")
        ed = arguments.get("end_time")
        t0, t1 = self._get_time_range(st, ed)

        wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.video_path,
                    "-ss",
                    str(max(0.0, t0)),
                    "-to",
                    str(min(float(self.duration), t1)),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    wav_path,
                ],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception as e:
            print(f"  ffmpeg audio extract failed: {e}")
            return {"query": query, "events": [], "audio_summary": "audio extract failed", "backend": "none"}

        if _CLAP_MODULE is None:
            _CLAP_MODULE = CLAP_Module(enable_fusion=False)
            _CLAP_MODULE.load_ckpt()

        audio_data, sr = sf.read(wav_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
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
                sf.write(chunk_wav, chunk, int(sr))
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

        try:
            os.unlink(wav_path)
        except OSError:
            pass

        return {
            "query": query,
            "events": events,
            "audio_summary": f"CLAP scan {len(events)} peaks; query={query!r}",
            "backend": "laion_clap",
        }

    def _process_audio_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "audio_grounder")
        if not calls:
            return ""

        print("\n[Tool] Audio Grounder")
        results = []

        for arguments in calls:
            result = None
            if os.getenv("REFINER_DISABLE_CLAP", "").strip() != "1":
                try:
                    result = self._audio_grounder_clap(arguments if isinstance(arguments, dict) else {})
                except Exception as e:
                    print(f"  LAION-CLAP fallback: {e}")
            if result is None or result.get("backend") == "none":
                asr_result = self._get_asr_result_from_subtitles(
                    arguments.get("start_time"), arguments.get("end_time")
                )
                summary = (
                    "Speech subtitles are available in this range, but non-speech audio grounding is unavailable."
                    if asr_result["segments"]
                    else "Non-speech audio grounding is unavailable in this runner."
                )
                result = {
                    "query": str(arguments.get("query", "")).strip(),
                    "events": [],
                    "audio_summary": summary,
                    "backend": "stub",
                }
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
            frame_path = arguments.get("frame_path")
            frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)
            return_masks = bool(arguments.get("return_masks", False))
            if not frame_path or not os.path.exists(frame_path):
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            backend = str(getattr(self, "spatial_grounder_backend", "grounding_dino") or "grounding_dino").strip().lower()
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
            frame_path = arguments.get("frame_path")
            frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)
            if not frame_path or not os.path.exists(frame_path):
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

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
            result = self._run_vlm_json(
                prompt,
                [frame_path] if frame_path else [],
                [float(frame_ts)],
                default_result,
            )
            results.append(self._format_refine_tool_result("counter", arguments, result))

        return "".join(results)

    def _process_dense_captioner(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "dense_captioner")
        if not calls:
            return ""

        print("\n[Tool] Dense Captioner")
        results = []

        for arguments in calls:
            granularity = str(arguments.get("granularity", "segment") or "segment")
            fps = 1.0 if granularity == "frame" else 2.0
            frame_paths, timestamps, start, end = self._get_frames_for_range(
                arguments.get("start_time"), arguments.get("end_time"), fps=fps
            )
            focus_query = str(arguments.get("focus_query", "")).strip()
            default_result = {
                "video_duration": float(self.duration),
                "captioned_range": {"start": start, "end": end},
                "captions": [],
                "overall_summary": "",
            }
            prompt = (
                dense_captioner_prompt.strip()
                + f"\n\nGranularity: {granularity}. Focus query: {focus_query}\nReturn JSON only.\n"
            )
            result = self._run_vlm_json(prompt, frame_paths, timestamps, default_result)
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

            try:
                prompt_text = chart_analyzer_prompt.strip()
                if len(frame_paths) > 1:
                    prompt_text += "\n\nUse all provided frames jointly as multiple retrieved views of the same chart.\n"
                if query:
                    prompt_text += f"\n\nQuery: {query}\nReturn JSON only matching the OUTPUT FORMAT above.\n"
                else:
                    prompt_text += "\n\nReturn JSON only matching the OUTPUT FORMAT above.\n"

                mode = getattr(self, "chart_mode", "api")
                raw_output = None
                parsed = None

                if mode == "api":
                    raw_output = self._call_chart_vision_api(prompt_text, frame_paths[0])
                    print(f'chart analyzer output: {raw_output}')
                    parsed = self._extract_json_payload(raw_output)
                elif mode == "vlm":
                    merged = self._run_vlm_json(
                        prompt_text, frame_paths, frame_timestamps, default_result
                    )
                    if isinstance(merged, dict) and "raw_output" in merged:
                        parsed = self._extract_json_payload(merged.get("raw_output", ""))
                    elif isinstance(merged, dict):
                        parsed = merged
                elif mode == "internvl":
                    self._load_chart_model()
                    pixel_values = self._internvl_load_image(frame_path)
                    generation_config = dict(max_new_tokens=1024, do_sample=False)
                    raw_output = self._chart_model.chat(
                        self._chart_tokenizer,
                        pixel_values,
                        prompt_text,
                        generation_config,
                    )
                    parsed = self._extract_json_payload(raw_output)
                else:
                    default_result["query_response"] = f"unknown chart_mode: {mode}"
                    results.append(self._format_refine_tool_result("chart_analyzer", arguments, default_result))
                    continue

                if isinstance(parsed, dict):
                    result = parsed
                    result.setdefault("query_response", None)
                else:
                    default_result["query_response"] = (raw_output or "").strip() if raw_output else ""
                    result = default_result
            except Exception as e:
                print(f"  Chart analyzer error: {e}")
                default_result["query_response"] = f"chart_analyzer error: {e}"
                result = default_result

            results.append(self._format_refine_tool_result("chart_analyzer", arguments, result))

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
