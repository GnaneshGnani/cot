import argparse
import base64
import fcntl
import hashlib
from html import parser
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

# PaddleOCR/PaddleX ship a vLLM plugin (register_paddlex_genai_models) that expects
# newer vLLM (e.g. Ernie4.5). vllm==0.8.4 has no vllm.model_executor.models.ernie45.
if "VLLM_PLUGINS" not in os.environ:
    os.environ["VLLM_PLUGINS"] = ""

import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    VideoFileClip = None

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import refiner_debug
from refiner_agents import RefinerAgentsMixin
from refiner_tools import RefinerToolsMixin
from refiner_utils import (
    RefinerUtilsMixin,
    openai_chat_completion_limit_kwargs,
    openai_chat_temperature_kwargs,
)
from video_utils import parse_subtitle_time

def safe_write_with_lock(data, file_path):
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def safe_read_with_lock(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return pickle.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f'Error reading file {file_path}: {e}')
        return None

def list_to_sha256(lst):
    json_str = json.dumps(lst, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _norm_api_list(val, default_if_none):
    if val is None:
        return list(default_if_none) if default_if_none is not None else []
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return [str(x).strip() for x in val if str(x).strip()]


def _load_annotations(annotation_path: Path):
    raw = annotation_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {annotation_path}")
    return data


class VideoQADemo(RefinerUtilsMixin, RefinerToolsMixin, RefinerAgentsMixin):
    def __init__(self,
                 video_path: str,
                 question: str,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 10,
                 use_subtitle: bool = True,
                 vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                 vlm_tensor_parallel_size: int = 1,
                 vlm_api_base=None,
                 vlm_api_keys=None,
                 planner_model_name: str = "deepseek-ai/DeepSeek-V3",
                 planner_api_base=None,
                 planner_api_keys=None,
                 verifier_model_name: str = None,
                 verifier_api_base=None,
                 verifier_api_keys=None,
                 chart_mode: str = "api",
                 chart_model_name: str = "gpt-5",
                 chart_device: str = "cuda:0",
                 asr_device: str = "cuda:0",
                 asr_compute_type: str = None,
                 chart_api_base=None,
                 chart_api_keys=None,
                 spatial_grounder_backend: str = "vlm",
                 spatial_grounder_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                 spatial_grounder_device: str = "cuda:0",
                 spatial_grounder_box_threshold: float = 0.25,
                 spatial_grounder_iou_threshold: float = 0.8,
                 spatial_grounder_vlm_fallback: bool = True,
                 refinement_debug_root: str = None,
                 dense_frame_fps: float = None,
                 use_clip_retrieval: bool = False,
                 dense_segment_half_width: float = 0.5,
                 retrieval_top_k: int = 5,
                 dense_frame_embed_batch: int = 8,
                 temporal_grounder_device_index: int = None):
        self.video_path = video_path
        self.question = question
        self.answer = answer
        self.options = options or []
        self.dataset_folder = dataset_folder
        self.clip_duration = clip_duration
        self.use_subtitle = use_subtitle
        self.use_clip_retrieval = use_clip_retrieval
        self.dense_segment_half_width = float(dense_segment_half_width)
        self.retrieval_top_k = int(retrieval_top_k)
        self.dense_frame_embed_batch = max(1, int(dense_frame_embed_batch))
        self.temporal_grounder_backend = (
            str(os.getenv("TEMPORAL_GROUNDER_BACKEND", "qwen")).strip().lower() or "qwen"
        )
        self.temporal_grounder_model_name = (
            str(
                os.getenv(
                    "TEMPORAL_GROUNDER_MODEL_NAME",
                    "Qwen/Qwen3-VL-Embedding-8B",
                )
            ).strip()
            or "Qwen/Qwen3-VL-Embedding-8B"
        )
        self.temporal_grounder_reranker_model_name = (
            str(
                os.getenv(
                    "TEMPORAL_GROUNDER_RERANKER_MODEL_NAME",
                    "Qwen/Qwen3-VL-Reranker-8B",
                )
            ).strip()
            or "Qwen/Qwen3-VL-Reranker-8B"
        )
        self.temporal_grounder_sample_fps = max(
            0.1,
            float(os.getenv("TEMPORAL_GROUNDER_SAMPLE_FPS", "1.0") or 1.0),
        )
        self.temporal_grounder_max_frames = max(
            1,
            int(os.getenv("TEMPORAL_GROUNDER_MAX_FRAMES", "32") or 32),
        )
        self.temporal_grounder_batch_size = max(
            1,
            int(
                os.getenv(
                    "TEMPORAL_GROUNDER_BATCH_SIZE",
                    "4",
                )
                or 4
            ),
        )
        self.temporal_grounder_stride_seconds = max(
            0.1,
            float(
                os.getenv(
                    "TEMPORAL_GROUNDER_STRIDE_SECONDS",
                    str(max(1.0, float(self.clip_duration) / 2.0)),
                )
                or max(1.0, float(self.clip_duration) / 2.0)
            ),
        )
        self._temporal_grounder_video_info_cache = None
        self._temporal_grounder_qwen_clip_embeddings_cache = None
        self._temporal_grounder_embedder_class = None
        self._temporal_grounder_reranker_class = None
        # GPU index for tool models (temporal grounder); kept off vLLM's GPU 0
        _tg_dev_env = str(os.environ.get("TEMPORAL_GROUNDER_DEVICE_INDEX", "") or "").strip()
        if temporal_grounder_device_index is not None:
            self.temporal_grounder_device_index = int(temporal_grounder_device_index)
        elif _tg_dev_env.isdigit():
            self.temporal_grounder_device_index = int(_tg_dev_env)
        # else: not set → _temporal_grounder_device_index() auto-selects GPU 1 if available

        self._setup_environment()

        self.vlm_model_name = vlm_model_name
        self.local_vlm_model_name = (
            str(os.environ.get("LOCAL_VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")).strip()
            or "Qwen/Qwen3-VL-8B-Instruct"
        )
        self.vlm_tensor_parallel_size = int(vlm_tensor_parallel_size)
        self.planner_model_name = planner_model_name
        self.chart_model_name = chart_model_name
        self.chart_device = chart_device
        self.asr_device = str(asr_device or "cuda:0").strip() or "cuda:0"
        self.asr_compute_type = None if asr_compute_type is None else str(asr_compute_type).strip()
        cm = (chart_mode or "api").strip().lower()
        if cm not in ("api", "vlm", "internvl"):
            raise ValueError(f"chart_mode must be 'api', 'vlm', or 'internvl', got {chart_mode!r}")
        self.chart_mode = cm
        sgb = (spatial_grounder_backend or "grounding_dino").strip().lower()
        if sgb in ("grounding-dino", "groundingdino", "gdino", "auto", "hf", "huggingface"):
            sgb = "grounding_dino"
        if sgb not in ("grounding_dino", "vlm"):
            raise ValueError(
                "spatial_grounder_backend must be 'grounding_dino' or 'vlm', "
                f"got {spatial_grounder_backend!r}"
            )
        self.spatial_grounder_backend = sgb
        self.spatial_grounder_model_name = (
            str(spatial_grounder_model_name or "IDEA-Research/grounding-dino-base").strip()
            or "IDEA-Research/grounding-dino-base"
        )
        self.spatial_grounder_device = str(spatial_grounder_device or "cuda:0").strip() or "cuda:0"
        self.spatial_grounder_box_threshold = float(spatial_grounder_box_threshold)
        self.spatial_grounder_iou_threshold = float(spatial_grounder_iou_threshold)
        self.spatial_grounder_vlm_fallback = bool(spatial_grounder_vlm_fallback)

        self.planner_api_base = _norm_api_list(planner_api_base, ["http://localhost:8000/v1"])
        self.planner_api_keys = _norm_api_list(planner_api_keys, ["EMPTY"])

        # Verifier model — defaults to planner when not explicitly set
        self.verifier_model_name = (verifier_model_name or planner_model_name)
        self.verifier_api_base = (
            _norm_api_list(verifier_api_base, None) if verifier_api_base is not None
            else list(self.planner_api_base)
        )
        self.verifier_api_keys = (
            _norm_api_list(verifier_api_keys, ["EMPTY"]) if verifier_api_keys is not None
            else list(self.planner_api_keys)
        )

        self.vlm_api_base = _norm_api_list(vlm_api_base, None)
        self.vlm_api_keys = (
            _norm_api_list(vlm_api_keys, ["EMPTY"])
            if vlm_api_keys is not None
            else ["EMPTY"]
        )

        # None = inherit from planner_api_* when chart_mode is api
        self.chart_api_base = (
            _norm_api_list(chart_api_base, None) if chart_api_base is not None else None
        )
        self.chart_api_keys = (
            _norm_api_list(chart_api_keys, ["EMPTY"]) if chart_api_keys is not None else None
        )

        self._chart_model = None
        self._chart_tokenizer = None
        self._dense_captioner_vlm_server = None
        self._dense_captioner_processor = None

        self.refinement_debug_root = refiner_debug.resolve_debug_root(refinement_debug_root)
        self._refinement_debug_session_base = None
        self._refinement_debug_iter_dir = None
        self._refinement_debug_vlm_outputs_dir = None
        self._refinement_debug_vlm_input_basename = None

        self._initialize_models()

        self.duration = self._get_video_duration()
        self._video_fps = self._get_video_fps()
        self._dense_frame_fps_override = (
            None if dense_frame_fps is None else float(dense_frame_fps)
        )
        self.dense_frame_fps = (
            self._dense_frame_fps_override
            if self._dense_frame_fps_override is not None
            else 1.0
        )

        self._ensure_video_clip_embeddings()

        self.subtitles = self._extract_subtitles()

        self.messages = []
        
        print(f"✓ Demo initialized successfully")
        print(f"  Video: {video_path}")
        print(f"  Duration: {self.duration}s")
        print(f"  Question: {question}")
        if self.subtitles:
            print(f"  Subtitles: {len(self.subtitles)} characters")

    def set_task(self, question: str, answer: str = None, options: list = None):
        self.question = (question or "").strip()
        self.answer = answer
        self.options = list(options or [])

    def _ensure_refinement_debug_session_base(self) -> str | None:
        if not self.refinement_debug_root:
            self._refinement_debug_session_base = None
            return None
        if self._refinement_debug_session_base:
            return self._refinement_debug_session_base

        stem = refiner_debug.sanitize_path_component(Path(self.video_path).stem)
        base = Path(self.refinement_debug_root) / stem
        unique_debug = os.environ.get("REFINER_DEBUG_UNIQUE_RUN", "1").strip() != "0"
        if unique_debug:
            rid = (os.environ.get("REFINER_DEBUG_RUN_ID") or "").strip() or time.strftime(
                "%Y%m%d_%H%M%S"
            )
            base = base / refiner_debug.sanitize_path_component(rid)
        base.mkdir(parents=True, exist_ok=True)
        self._refinement_debug_session_base = str(base.resolve())
        return self._refinement_debug_session_base

    def load_sample(self, video_path: str, question: str, answer: str = None, options: list = None):
        # Release the resident frame embedder so the new video's frames are
        # re-embedded with a fresh cache on first frame_retriever call.
        self._release_frame_embedder()
        self.video_path = str(video_path)
        self.set_task(question, answer=answer, options=options)
        self._temporal_grounder_video_info_cache = None
        self._temporal_grounder_qwen_clip_embeddings_cache = None
        self.duration = self._get_video_duration()
        self._video_fps = self._get_video_fps()
        self.dense_frame_fps = (
            self._dense_frame_fps_override
            if self._dense_frame_fps_override is not None
            else 1.0
        )
        self._ensure_video_clip_embeddings()
        self.subtitles = self._extract_subtitles()

        print("✓ Sample loaded")
        print(f"  Video: {video_path}")
        print(f"  Duration: {self.duration}s")
        print(f"  Question: {question}")
        if self.subtitles:
            print(f"  Subtitles: {len(self.subtitles)} characters")
    
    def _setup_environment(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        torch.backends.cuda.matmul.allow_tf32 = True

    def _use_vlm_remote_api(self) -> bool:
        b = self.vlm_api_base
        return bool(b) and bool((b[0] or "").strip())

    def _resolve_local_vlm_model_path(self, model_name: str) -> str:
        raw_name = str(model_name or "").strip()
        if not raw_name:
            raise RuntimeError("LOCAL_VLM_MODEL_NAME is empty.")

        direct_path = Path(raw_name).expanduser()
        if direct_path.exists():
            return str(direct_path.resolve())

        if "/" not in raw_name:
            raise RuntimeError(
                f"Local VLM model {raw_name!r} is neither an existing path nor a Hugging Face repo id."
            )

        hf_home = str(os.environ.get("HF_HOME", "")).strip()
        if not hf_home:
            raise RuntimeError(
                f"HF_HOME is unset, so the local cache for {raw_name!r} cannot be resolved."
            )

        repo_dir = Path(hf_home) / "hub" / f"models--{raw_name.replace('/', '--')}"
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

        raise RuntimeError(
            f"No local snapshot found for {raw_name!r} under {repo_dir}. "
            "Download the model locally or set LOCAL_VLM_MODEL_NAME to a complete local path."
        )

    def _validate_local_vlm_model_path(self, resolved_path: str, requested_name: str) -> None:
        model_dir = Path(resolved_path)

        if not (model_dir / "config.json").is_file():
            raise RuntimeError(
                f"Local VLM model {requested_name!r} at {resolved_path} is missing config.json."
            )

        processor_files = ("processor_config.json", "preprocessor_config.json")
        if not any((model_dir / name).is_file() for name in processor_files):
            raise RuntimeError(
                f"Local VLM model {requested_name!r} at {resolved_path} is missing processor files "
                f"({', '.join(processor_files)})."
            )

        tokenizer_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
        )
        if not any((model_dir / name).is_file() for name in tokenizer_files):
            raise RuntimeError(
                f"Local VLM model {requested_name!r} at {resolved_path} is missing tokenizer assets "
                f"({', '.join(tokenizer_files)})."
            )

        weight_globs = (
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.gguf",
        )
        has_weights = any(any(model_dir.glob(pattern)) for pattern in weight_globs)
        if not has_weights:
            raise RuntimeError(
                f"Local VLM model {requested_name!r} at {resolved_path} is missing model weights."
            )

    def _load_local_vlm_runtime(self, requested_model_name: str):
        resolved_local_model = self._resolve_local_vlm_model_path(requested_model_name)
        self._validate_local_vlm_model_path(resolved_local_model, requested_model_name)

        _mm_kw = {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        }
        vlm_server = LLM(
            model=resolved_local_model,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=self.vlm_tensor_parallel_size,
            max_model_len=32768,
            enable_chunked_prefill=True,
            enforce_eager=True,
            mm_processor_kwargs=_mm_kw,
        )

        processor = AutoProcessor.from_pretrained(
            resolved_local_model,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "left"
        return vlm_server, processor, resolved_local_model

    def _local_vlm_sampling_params(self) -> SamplingParams:
        max_tokens_raw = str(os.environ.get("REFINER_VLM_MAX_TOKENS", "512")).strip()
        try:
            max_tokens = max(64, int(max_tokens_raw))
        except Exception:
            max_tokens = 512
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def _max_vlm_sequence_images(self) -> int:
        raw = str(os.environ.get("REFINER_VLM_MAX_IMAGES", "32")).strip()
        try:
            value = int(raw)
        except Exception:
            value = 32
        return max(1, value)

    def _uniform_subsample_sequence(self, image_paths: list, timestamps: list, max_items: int):
        image_paths = list(image_paths or [])
        timestamps = list(timestamps or [])
        total = len(image_paths)
        if total <= max_items:
            return image_paths, timestamps

        if len(timestamps) < total:
            timestamps = timestamps + [None] * (total - len(timestamps))

        if max_items <= 1:
            indices = [total // 2]
        else:
            indices = []
            for i in range(max_items):
                raw_idx = int(round(i * (total - 1) / float(max_items - 1)))
                min_idx = indices[-1] + 1 if indices else 0
                max_idx = total - (max_items - i)
                idx = min(max(raw_idx, min_idx), max_idx)
                indices.append(idx)

        return [image_paths[i] for i in indices], [timestamps[i] for i in indices]

    def _load_vlm_images(self, image_paths: list, timestamps: list):
        valid_paths = []
        valid_timestamps = []
        image_data = []

        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path)
                image.verify()
                image = Image.open(img_path)

                width, height = image.size
                if max(width, height) > 768:
                    if width > height:
                        new_width = 768
                        new_height = int(height * (768 / width))
                    else:
                        new_height = 768
                        new_width = int(width * (768 / height))
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                image_data.append(image)
                valid_paths.append(img_path)
                valid_timestamps.append(timestamps[idx] if idx < len(timestamps) else None)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        return valid_paths, valid_timestamps, image_data

    def _ensure_dense_captioner_local_runtime(self):
        if self._dense_captioner_vlm_server is not None and self._dense_captioner_processor is not None:
            return
        if (
            not self._use_vlm_remote_api()
            and self.vlm_server is not None
            and self.processor is not None
            and self.vlm_model_name == self.local_vlm_model_name
        ):
            self._dense_captioner_vlm_server = self.vlm_server
            self._dense_captioner_processor = self.processor
            return

        print(f"Initializing local dense-captioner VLM model: {self.local_vlm_model_name}")
        server, processor, resolved_local_model = self._load_local_vlm_runtime(self.local_vlm_model_name)
        self._dense_captioner_vlm_server = server
        self._dense_captioner_processor = processor
        print(
            f"✓ Dense-captioner local VLM loaded: {self.local_vlm_model_name} "
            f"({resolved_local_model})"
        )

    def _initialize_models(self):
        if self._use_vlm_remote_api():
            self.vlm_server = None
            self.processor = None
            print(f"✓ VLM tools use remote API (model={self.vlm_model_name})")
            return

        print(f"Initializing local VLM model: {self.vlm_model_name}")
        self.vlm_server, self.processor, resolved_local_model = self._load_local_vlm_runtime(
            self.vlm_model_name
        )
        print(f"✓ VLM model loaded locally: {self.vlm_model_name} ({resolved_local_model})")

    def _vlm_vision_api_call(self, prompt: str, image_paths: list) -> str:
        content = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path)
                image.verify()
                image = Image.open(img_path)
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
                print(f"Error loading image {img_path}: {e}")

        if not content:
            return "Error: No valid frames"

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        vlm_out = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
        if vlm_out:
            fname = getattr(self, "_refinement_debug_vlm_input_basename", None) or "model_input.json"
            refiner_debug.write_json(
                vlm_out,
                fname,
                {"model": self.vlm_model_name, "messages": messages, "video_frame_paths": list(image_paths)},
            )

        pairs = list(zip(self.vlm_api_base, self.vlm_api_keys))
        if not pairs:
            return ""

        for base, key in pairs:
            try:
                client = OpenAI(base_url=base.strip(), api_key=key.strip())
                completion = client.chat.completions.create(
                    model=self.vlm_model_name,
                    messages=messages,
                    **openai_chat_temperature_kwargs(self.vlm_model_name, 0.01),
                    **openai_chat_completion_limit_kwargs(self.vlm_model_name, 2048),
                )
                out = completion.choices[0].message.content
                text = out if isinstance(out, str) else (out or "")
                if vlm_out:
                    refiner_debug.write_text(vlm_out, "vlm_raw_output.txt", text)
                return text.strip()
            except Exception as e:
                print(f"[VLM_VISION_API] ERROR base={base} model={self.vlm_model_name}: {e}")

        return ""
    
    def _ensure_retriever_ready(self) -> bool:
        try:
            self._ensure_video_clip_embeddings()
            return True
        except Exception as e:
            print(f"Warning: Qwen retriever preparation failed: {e}")
            return False

    def _ensure_video_clip_embeddings(self):
        # The active retrieval stack is Qwen-based. Pre-build dense frames and
        # their frame embeddings so frame_retriever does not depend on
        # LanguageBind clip preprocessing.
        self._ensure_dense_frames()

    def _ensure_dense_frames(self):
        """Pre-extract dense frames at dense_frame_fps so frame_retriever has them ready."""
        from video_utils import timestamp_to_clip_path
        video_id = Path(self.video_path).stem
        dense_dir = Path(self.dataset_folder) / "dense_frames" / video_id
        # Check if already populated
        existing = list(dense_dir.glob("frame_*.png")) if dense_dir.is_dir() else []
        if not existing:
            fps = float(getattr(self, "dense_frame_fps", 1.0))
            print(f"  Extracting dense frames at {fps} fps → {dense_dir} ...")
            try:
                timestamp_to_clip_path(
                    self.dataset_folder,
                    0.0,
                    float(self.duration),
                    self.video_path,
                    fps=fps,
                )
            except Exception as e:
                print(f"  Warning: dense frame extraction failed: {e}")

        # Pre-build Qwen3 frame embedding cache so frame_retriever calls only
        # need 1 forward pass (query) instead of re-embedding all frames each time.
        frame_paths = sorted(
            (dense_dir.glob("frame_*.png") if dense_dir.is_dir() else []),
            key=lambda p: p.name,
        )
        if frame_paths:
            frame_items = []
            for fp in frame_paths:
                try:
                    ts = float(fp.stem.replace("frame_", ""))
                except ValueError:
                    ts = 0.0
                frame_items.append({"frame_path": str(fp), "timestamp": ts})
            try:
                self._precompute_frame_embeddings_cache(frame_items)
            except Exception as e:
                print(f"  Warning: frame embedding precompute failed: {e}")
            # Ensure the embedder is warm in GPU memory now (cache hit path skips
            # embedding but we still need the model resident before tool calls).
            try:
                self._get_or_load_frame_embedder()
            except Exception as e:
                print(f"  Warning: frame embedder warm-up failed: {e}")
    
    def _get_video_duration(self):
        try:
            if VideoFileClip is None:
                raise RuntimeError("moviepy not available")
            with VideoFileClip(self.video_path) as video:
                return int(video.duration)
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            return 300

    def _get_video_fps(self) -> float:
        try:
            if VideoFileClip is None:
                raise RuntimeError("moviepy not available")
            with VideoFileClip(self.video_path) as video:
                fps = float(getattr(video, "fps", None) or 24.0)
                return max(1.0, min(fps, 120.0))
        except Exception as e:
            print(f"Warning: Could not get video FPS: {e}")
            return 24.0
    
    def _extract_subtitles(self):
        if not self.use_subtitle:
            return ""
        
        video_id = Path(self.video_path).stem
        subtitle_path = f'{self.dataset_folder}/subtitles/{video_id}.srt'
        
        if not os.path.exists(subtitle_path):
            print("No subtitle file found")
            return ""
        
        subtitles = ""
        try:
            with open(subtitle_path, "r", encoding="utf-8") as f:
                content = f.read().split("\n\n")
                for section in content:
                    if section.strip():
                        lines = section.split("\n")
                        if len(lines) >= 3:
                            time_range = lines[1].split(" --> ")
                            start_time = parse_subtitle_time(time_range[0])
                            end_time = parse_subtitle_time(time_range[1])
                            text = " ".join(lines[2:])
                            subtitles += f"{int(start_time)}-{int(end_time)}:{text} "
        except Exception as e:
            print(f"Error extracting subtitles: {e}")
            return ""
        
        return subtitles

    @staticmethod
    def _use_openai_http(api_base: list) -> bool:
        if os.environ.get("REFINER_FORCE_HTTP_LLM", "").strip() == "1":
            return True
        if os.environ.get("REFINER_USE_VLLM_PICKLE", "").strip() == "1":
            return False
        if not api_base:
            return False
        b = (api_base[0] or "").strip().lower()
        if not b:
            return False
        return "localhost" not in b and "127.0.0.1" not in b

    def _text2text(self, message: list, model_name: str, api_base: list, api_keys: list) -> str:
        print("\n" + "=" * 70)
        print("message: ", message)
        print("model_name: ", model_name)
        print("api_base: ", api_base)
        print("api_keys: ", api_keys)
        print("=" * 70 + "\n")

        if self._use_openai_http(api_base):
            normalized_messages = []
            for m in (message or []):
                if not isinstance(m, dict):
                    continue
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if not isinstance(content, str):
                    content = str(content)
                normalized_messages.append(
                    {"role": str(m.get("role", "user") or "user"), "content": content}
                )

            if not normalized_messages:
                print(f"[TEXT2TEXT] ERROR: empty normalized messages for model {model_name}")
                return ""

            pairs = list(zip(api_base, api_keys))
            if not pairs:
                print(f"[TEXT2TEXT] ERROR: no api base/key for model {model_name}")
                return ""

            for base, key in pairs:
                try:
                    client = OpenAI(base_url=base.strip(), api_key=key.strip())
                    request_kwargs = {
                        "model": model_name,
                        "messages": normalized_messages,
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
                    print(f"[TEXT2TEXT] ERROR base={base} model={model_name}: {e}")

            return ""

        folder_path = "_planner"
        start_time = time.time()

        index = message + [model_name] + [len(message)]
        file_name = f'{list_to_sha256(index)}.pkl'
        read_file = f'./vllm_io_files/vllm_input{folder_path}/{file_name}'
        safe_write_with_lock({'model': model_name, 'input': message},read_file)
        start_time = time.time()
        while True:
            output_file = f'./vllm_io_files/vllm_output{folder_path}/{file_name}'
            if os.path.exists(output_file):
                end_time = time.time()
                try:
                    ans = safe_read_with_lock(output_file)
                    return ans
                except Exception as e:
                    print('[TEXT2TEXT] ERROR:', e)
                    safe_write_with_lock({'model': model_name, 'input': message},read_file)
                    os.system(f'rm {output_file}')

            if time.time()-start_time>120:
                break
            if not os.path.exists(read_file):
                safe_write_with_lock({'model': model_name, 'input': message},read_file)
            time.sleep(0.2)

        print('[TEXT2TEXT] ERROR: Timeout, model:', model_name)
        return ''
    
    
    def _batch_video2text(self, tasks: list, force_local: bool = False):
        results = []

        for prompt, image_paths, timestamps in tasks:
            if not force_local and self._use_vlm_remote_api():
                result = self._vlm_vision_api_call(prompt, image_paths)
                results.append(result)
                continue

            if force_local:
                self._ensure_dense_captioner_local_runtime()
                active_server = self._dense_captioner_vlm_server
                active_processor = self._dense_captioner_processor
                active_model_name = self.local_vlm_model_name
            else:
                active_server = self.vlm_server
                active_processor = self.processor
                active_model_name = self.vlm_model_name

            selected_paths = list(image_paths or [])
            selected_timestamps = list(timestamps or [])
            original_count = len(selected_paths)
            max_sequence_images = self._max_vlm_sequence_images()
            if len(selected_paths) > max_sequence_images:
                print(
                    f"Reducing multi-frame VLM request from {len(selected_paths)} to "
                    f"{max_sequence_images} evenly spaced frames before generation."
                )
                selected_paths, selected_timestamps = self._uniform_subsample_sequence(
                    selected_paths, selected_timestamps, max_sequence_images
                )

            valid_paths, valid_timestamps, image_data = self._load_vlm_images(
                selected_paths, selected_timestamps
            )
            if not image_data:
                results.append("Error: No valid frames")
                continue

            sampling_params = self._local_vlm_sampling_params()
            if len(image_data) > 1:
                token_budget = max(1, 32768 - int(getattr(sampling_params, "max_tokens", 512) or 512) - 1024)
                max_image_tokens = max(
                    max(1, ((img.size[0] + 27) // 28) * ((img.size[1] + 27) // 28))
                    for img in image_data
                )
                allowed_images = max(1, min(max_sequence_images, token_budget // max_image_tokens))
                if len(valid_paths) > allowed_images:
                    print(
                        f"Reducing multi-frame VLM request from {len(valid_paths)} to "
                        f"{allowed_images} evenly spaced frames to fit model context."
                    )
                    valid_paths, valid_timestamps = self._uniform_subsample_sequence(
                        valid_paths, valid_timestamps, allowed_images
                    )
                    valid_paths, valid_timestamps, image_data = self._load_vlm_images(
                        valid_paths, valid_timestamps
                    )
                    if not image_data:
                        results.append("Error: No valid frames")
                        continue

            single_frame = len(image_data) == 1
            content = []
            if single_frame:
                content.append({"type": "image", "image": valid_paths[0]})
            else:
                # Local Qwen3-VL expects real video metadata for `video` inputs.
                # Our tool pipeline provides pre-extracted frames, so send them as
                # an image sequence instead of a synthetic video payload.
                for img_path in valid_paths:
                    content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            formatted_prompt = active_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            mm_kwargs = {
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 768 * 28 * 28,
            }
            vlm_out = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
            if vlm_out:
                fname = getattr(self, "_refinement_debug_vlm_input_basename", None) or "model_input.json"
                refiner_debug.write_json(
                    vlm_out,
                    fname,
                    {
                        "model": active_model_name,
                        "formatted_prompt": formatted_prompt,
                        "video_frame_paths": list(valid_paths),
                        "timestamps": list(valid_timestamps),
                        "input_mode": "image" if single_frame else "image_sequence",
                        "mm_processor_kwargs": dict(mm_kwargs),
                        "messages": messages,
                        "original_image_count": original_count,
                        "used_image_count": len(valid_paths),
                    },
                )

            outputs = active_server.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": image_data[0]} if single_frame else {"image": image_data},
                    "mm_processor_kwargs": mm_kwargs,
                },
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            result = ((outputs[0].outputs[0].text or "") if outputs and outputs[0].outputs else "").strip()
            if vlm_out:
                refiner_debug.write_text(vlm_out, "vlm_raw_output.txt", result)
            results.append(result)
        
        return results

    def run_refinement_pipeline(self, trace_steps: list = None, trace_answer: str = None, max_iterations: int = 1):
        print("\n" + "=" * 70)
        print("Starting Trace Refinement Pipeline")
        print("=" * 70 + "\n")

        generated_trace_info = None
        trace_steps = list(trace_steps or [])

        # Cold-start generation: produce a trace from scratch if none provided
        if not trace_steps:
            gen_steps, gen_answer, gen_rounds = self._call_trace_generator()
            trace_steps = gen_steps
            trace_answer = gen_answer
            generated_trace_info = {
                "trace_steps": gen_steps,
                "answer": gen_answer,
                "generation_rounds": gen_rounds,
            }
            print("\n" + "=" * 70)
            print(f"Trace generated: {len(gen_steps)} steps, answer: {gen_answer}")
            print("=" * 70 + "\n")

        trace_answer = (trace_answer or self._extract_trace_answer(trace_steps) or "").strip()
        question_text = self._format_question_with_options()
        initial_trace = list(trace_steps)
        initial_answer = trace_answer
        current_trace = list(trace_steps)
        current_answer = trace_answer
        iteration_history = []
        all_iterations = []

        print("\n" + "=" * 70)
        print("trace_answer: ", trace_answer)
        print("=" * 70 + "\n")

        final_verifier_raw = None
        final_verifier_output = None

        debug_resolved = self._ensure_refinement_debug_session_base()

        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")

            if self._refinement_debug_session_base:
                self._refinement_debug_iter_dir = str(
                    Path(self._refinement_debug_session_base) / f"iteration_{iteration + 1}"
                )
                Path(self._refinement_debug_iter_dir).mkdir(parents=True, exist_ok=True)
            else:
                self._refinement_debug_iter_dir = None

            print("[Verifier] Generating diagnosis...")
            verifier_raw, verifier_output = self._call_verifier(
                current_trace,
                current_answer,
                question_text=question_text,
                iteration=iteration,
                history=iteration_history,
                max_iterations=max_iterations,
            )
            final_verifier_raw, final_verifier_output = verifier_raw, verifier_output
            print(f"\n[Verifier Output]\n{verifier_raw}\n")

            if isinstance(verifier_output, dict) and verifier_output.get("verdict") == "PASS":
                print("[Verifier] PASS — stopping refinement loop.")
                break

            print("[Planner] Generating plan...")
            planner_raw, planner_output = self._call_planner(
                current_trace,
                current_answer,
                verifier_output if verifier_output is not None else verifier_raw,
                iteration=iteration,
                history=iteration_history,
                max_iterations=max_iterations,
            )
            print(f"\n[Planner Output]\n{planner_raw}\n")

            print("[Executor] Running planned tool calls...")
            executed_tools = self._execute_refine_plan(planner_output if planner_output is not None else {})
            for item in executed_tools:
                print(f"  Step {item['step']} - {item['tool']}")

            print("[Refiner] Synthesizing corrected trace...")
            refiner_raw, refiner_output = self._call_refiner(
                current_trace,
                current_answer,
                verifier_output if verifier_output is not None else verifier_raw,
                executed_tools,
                planner_output if planner_output is not None else {},
            )
            print(f"\n[Refiner Output]\n{refiner_raw}\n")

            if isinstance(refiner_output, dict):
                new_trace = refiner_output.get("refined_trace", current_trace)
                new_answer = refiner_output.get("refined_answer", current_answer)
                current_trace = self._normalize_refined_trace(new_trace, current_trace)
                if new_answer is not None and str(new_answer).strip():
                    current_answer = str(new_answer).strip()

            summary = self._compact_iteration_summary(
                iteration, verifier_output, refiner_output, executed_tools
            )
            iteration_history.append(summary)
            all_iterations.append(
                {
                    "iteration": iteration + 1,
                    "verifier_raw": verifier_raw,
                    "verifier_output": verifier_output,
                    "planner_raw": planner_raw,
                    "planner_output": planner_output,
                    "executed_tools": executed_tools,
                    "refiner_raw": refiner_raw,
                    "refiner_output": refiner_output,
                    "iteration_summary": summary,
                }
            )

        self._refinement_debug_iter_dir = None

        # Release the persistent frame embedder now that the pipeline is done.
        # This frees the ~16 GB it occupies on GPU 1.
        self._release_frame_embedder()

        # Resolve bare MCQ letters (e.g. "A") to full option text so that
        # downstream comparisons work correctly for MCQ questions.
        resolved_answer = self._resolve_mcq_answer(current_answer)
        is_correct = self._answers_match(resolved_answer, self.answer or "") if self.answer else None

        return {
            "question": self.question,
            "options": self.options,
            "video_path": self.video_path,
            "trace_generated": generated_trace_info is not None,
            "generated_trace": generated_trace_info,
            "initial_trace": {"steps": initial_trace},
            "initial_answer": initial_answer,
            "final_trace": {"steps": current_trace},
            "final_answer": resolved_answer,
            "final_answer_raw": current_answer,
            "is_correct": is_correct,
            "verifier_raw": final_verifier_raw,
            "verifier_output": final_verifier_output,
            "iteration_history": iteration_history,
            "all_iterations": all_iterations,
            "max_iterations": max_iterations,
            "refinement_debug_root": debug_resolved,
        }


def main():
    # default_annotation_file = "/nfs-stor/ghazi.ahmad/videos/annotations.json"
    default_annotation_file = os.path.join(_eval_dir, "refiner_inputs.json")

    def _env_list(name: str, fallback: str = ""):
        raw = (os.environ.get(name, fallback) or "").strip()
        if not raw:
            return None
        return [item.strip() for item in raw.split(",") if item.strip()]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "annotation_file",
        nargs="?",
        default=default_annotation_file,
        help="Path to annotations.json (or JSONL)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max refinement iterations per sample")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of a single entry to process (0-based); omit to process all entries",
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotation_file).expanduser().resolve()
    if not annotation_path.exists():
        raise SystemExit(f"Error: annotation file not found: {annotation_path}")

    results_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(_eval_dir) / "results_generated"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    data = _load_annotations(annotation_path)
    if not data:
        raise SystemExit(f"Error: no samples found in {annotation_path}")
    if args.index is not None:
        if args.index < 0 or args.index >= len(data):
            raise SystemExit(f"Error: --index {args.index} out of range (0..{len(data) - 1})")
        data = [data[args.index]]

    openai_api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "").strip()
    planner_api_base = _env_list(
        "PLANNER_API_BASE", os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )
    planner_api_keys = _env_list("PLANNER_API_KEY", os.environ.get("API_KEY", openai_api_key))
    if not planner_api_keys:
        raise SystemExit("Error: set PLANNER_API_KEY, API_KEY, or OPENAI_API_KEY before running.")

    vlm_api_base = _env_list("VLM_API_BASE")
    vlm_api_keys = _env_list("VLM_API_KEY", openai_api_key) if vlm_api_base else None
    if vlm_api_base and not vlm_api_keys:
        raise SystemExit("Error: set VLM_API_KEY, API_KEY, or OPENAI_API_KEY before running.")
    local_vlm_model_name = os.environ.get("LOCAL_VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    default_remote_vlm_model = os.environ.get("API_MODEL_NAME", "gpt-5.4")
    if vlm_api_base:
        vlm_model_name = os.environ.get("VLM_MODEL_NAME", default_remote_vlm_model)
    else:
        vlm_model_name = os.environ.get("VLM_MODEL_NAME", local_vlm_model_name)

    chart_mode = os.environ.get("CHART_MODE", "vlm")
    chart_model_name = os.environ.get("CHART_MODEL_NAME", vlm_model_name)
    planner_model_name = os.environ.get("PLANNER_MODEL_NAME", os.environ.get("API_MODEL_NAME", "gpt-5.4"))

    verifier_model_name = os.environ.get("VERIFIER_MODEL_NAME") or None
    verifier_api_base = _env_list("VERIFIER_API_BASE") or None
    verifier_api_keys = _env_list("VERIFIER_API_KEY") or None

    _tg_dev_env = os.environ.get("TEMPORAL_GROUNDER_DEVICE_INDEX", "").strip()
    temporal_grounder_device_index = int(_tg_dev_env) if _tg_dev_env.isdigit() else None

    if vlm_api_base:
        print(f"Using remote VLM API: model={vlm_model_name} base={vlm_api_base[0]}")
    else:
        print(f"Using local VLM model: {vlm_model_name}")

    demo = None
    dataset_folder = str((Path(_eval_dir) / "data").resolve())
    debug_root = str((Path(_eval_dir) / "debug").resolve())

    def _save_result(record, video_path):
        video_stem = Path(video_path).stem if video_path else "unknown"
        out_file = results_dir / f"{video_stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return out_file

    for index, item in enumerate(data[23:24], start=1):
        record = dict(item)
        video_path = str(item.get("video_path", "")).strip()
        question = str(item.get("question", "")).strip()
        options = list(item.get("options") or [])
        trace_steps = []
        input_answer = item.get("answer")
        # Always generate the initial trace from scratch without exposing the gold
        # answer to the pipeline. We still score against the input answer after the run.
        pipeline_answer = None

        print(f"\n[{index}/{len(data)}] {Path(video_path).name or '<missing video>'}")
        if item.get("initial_trace_steps"):
            print("  Ignoring existing initial_trace_steps; generating a fresh trace.")

        if not video_path or not os.path.exists(video_path):
            record["refiner_error"] = f"Video file not found: {video_path}"
            _save_result(record, video_path)
            continue

        try:
            if demo is None:
                demo = VideoQADemo(
                    video_path=video_path,
                    question=question,
                    answer=pipeline_answer,
                    options=options,
                    dataset_folder=dataset_folder,
                    use_subtitle=False,
                    refinement_debug_root=debug_root,
                    dense_frame_fps=1.0,
                    use_clip_retrieval=False,
                    dense_segment_half_width=0.5,
                    retrieval_top_k=10,
                    dense_frame_embed_batch=8,
                    vlm_model_name=vlm_model_name,
                    vlm_api_base=vlm_api_base,
                    vlm_api_keys=vlm_api_keys,
                    planner_model_name=planner_model_name,
                    planner_api_base=planner_api_base,
                    planner_api_keys=planner_api_keys,
                    verifier_model_name=verifier_model_name,
                    verifier_api_base=verifier_api_base,
                    verifier_api_keys=verifier_api_keys,
                    chart_mode=chart_mode,
                    chart_model_name=chart_model_name,
                    temporal_grounder_device_index=temporal_grounder_device_index,
                )
            else:
                demo.load_sample(
                    video_path=video_path,
                    question=question,
                    answer=pipeline_answer,
                    options=options,
                )

            record["refiner_result"] = demo.run_refinement_pipeline(
                trace_steps=trace_steps if trace_steps else None,
                max_iterations=args.max_iterations,
            )
            if pipeline_answer is None and input_answer:
                final_answer = record["refiner_result"].get("final_answer") or record["refiner_result"].get("final_answer_raw", "")
                record["refiner_result"]["is_correct"] = demo._answers_match(final_answer, input_answer, options=options)
        except Exception as e:
            import traceback
            traceback.print_exc()
            record["refiner_error"] = str(e)

        out_file = _save_result(record, video_path)
        print(f"  ✓ Saved: {out_file}")

    print(f"\n✓ Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
