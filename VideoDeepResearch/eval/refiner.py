import base64
import fcntl
import hashlib
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
from retriever_languagebind import Retrieval_Manager
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


class VideoQADemo(RefinerUtilsMixin, RefinerToolsMixin, RefinerAgentsMixin):
    def __init__(self,
                 video_path: str,
                 question: str,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 5,
                 use_subtitle: bool = True,
                 vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 vlm_tensor_parallel_size: int = 1,
                 vlm_api_base=None,
                 vlm_api_keys=None,
                 planner_model_name: str = "deepseek-ai/DeepSeek-V3",
                 planner_api_base=None,
                 planner_api_keys=None,
                 chart_mode: str = "api",
                 chart_model_name: str = "gpt-5",
                 chart_device: str = "cuda:0",
                 chart_api_base=None,
                 chart_api_keys=None,
                 spatial_grounder_backend: str = "grounding_dino",
                 spatial_grounder_model_name: str = "IDEA-Research/grounding-dino-base",
                 spatial_grounder_device: str = "cuda:0",
                 spatial_grounder_box_threshold: float = 0.25,
                 spatial_grounder_iou_threshold: float = 0.8,
                 spatial_grounder_vlm_fallback: bool = True,
                 refinement_debug_root: str = None,
                 dense_frame_fps: float = None,
                 use_clip_retrieval: bool = False,
                 dense_segment_half_width: float = 0.5,
                 retrieval_top_k: int = 5,
                 dense_frame_embed_batch: int = 8):
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

        self._setup_environment()

        self.vlm_model_name = vlm_model_name
        self.vlm_tensor_parallel_size = int(vlm_tensor_parallel_size)
        self.planner_model_name = planner_model_name
        self.chart_model_name = chart_model_name
        self.chart_device = chart_device
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

        self.refinement_debug_root = refiner_debug.resolve_debug_root(refinement_debug_root)
        self._refinement_debug_session_base = None
        self._refinement_debug_iter_dir = None
        self._refinement_debug_vlm_outputs_dir = None
        self._refinement_debug_vlm_input_basename = None

        self._initialize_models()

        self.duration = self._get_video_duration()
        self._video_fps = self._get_video_fps()
        self.dense_frame_fps = (
            float(dense_frame_fps) if dense_frame_fps is not None else float(self._video_fps)
        )

        self.retriever = self._initialize_retriever()

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
        self.messages = []
    
    def _setup_environment(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        torch.backends.cuda.matmul.allow_tf32 = True

    def _use_vlm_remote_api(self) -> bool:
        b = self.vlm_api_base
        return bool(b) and bool((b[0] or "").strip())

    def _initialize_models(self):
        if self._use_vlm_remote_api():
            self.vlm_server = None
            self.processor = None
            print(f"✓ VLM tools use remote API (model={self.vlm_model_name})")
            return

        print("Initializing VLM model...")

        _mm_kw = {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        }
        self.vlm_server = LLM(
            model=self.vlm_model_name,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=self.vlm_tensor_parallel_size,
            max_model_len=32768,
            enable_chunked_prefill=True,
            enforce_eager=True,
            mm_processor_kwargs=_mm_kw,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"

        print(f"✓ VLM model loaded: {self.vlm_model_name}")

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
    
    def _initialize_retriever(self):
        print("Initializing retriever...")

        class Args:
            dataset_folder = self.dataset_folder
            dataset = "demo"
            clip_duration = self.clip_duration
            retriever_type = "large"
            clip_fps=2.0
        
        args = Args()
        clip_save_folder = f'{self.dataset_folder}/clips/{self.clip_duration}/'
        
        retriever = Retrieval_Manager(args, clip_save_folder=clip_save_folder)
        
        if torch.cuda.is_available():
            retriever.load_model_to_gpu(0)
        
        print(f"✓ Retriever initialized")
        return retriever
    
    def _ensure_video_clip_embeddings(self):
        folder_path = f'{self.dataset_folder}/embeddings/{self.clip_duration}/large'
        video_clip_paths, _ = self.retriever.calculate_video_clip_embedding(
            self.video_path, folder_path, total_duration=self.duration, pre_calculate=False
        )
        if len(video_clip_paths) == 0:
            print("Clip embeddings not found, preprocessing current video...")
            self.retriever.calculate_video_clip_embedding(
                self.video_path, folder_path, total_duration=self.duration, pre_calculate=True
            )
    
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
            for m in message:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if not isinstance(content, str):
                    content = str(content)
                normalized_messages.append({"role": m["role"], "content": content})

            pairs = list(zip(api_base, api_keys))
            if not pairs:
                print(f"[TEXT2TEXT] ERROR: no api base/key for model {model_name}")
                return ""

            for base, key in pairs:
                try:
                    client = OpenAI(base_url=base.strip(), api_key=key.strip())
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=normalized_messages,
                    )
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
    
    
    def _batch_video2text(self, tasks: list):
        results = []

        for prompt, image_paths, timestamps in tasks:
            if self._use_vlm_remote_api():
                result = self._vlm_vision_api_call(prompt, image_paths)
                results.append(result)
                continue

            image_data = []
            for img_path in image_paths:
                if os.path.exists(img_path):
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
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        continue
            
            if not image_data:
                results.append("Error: No valid frames")
                continue

            content = [
                {"type": "video", "video": image_paths},
                {"type": "text", "text": prompt}
            ]
            messages = [{"role": "user", "content": content}]

            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            fps = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 2.0
            
            mm_kwargs = {
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 768 * 28 * 28,
                "fps": fps,
            }
            vlm_out = getattr(self, "_refinement_debug_vlm_outputs_dir", None)
            if vlm_out:
                fname = getattr(self, "_refinement_debug_vlm_input_basename", None) or "model_input.json"
                refiner_debug.write_json(
                    vlm_out,
                    fname,
                    {
                        "formatted_prompt": formatted_prompt,
                        "video_frame_paths": list(image_paths),
                        "timestamps": list(timestamps),
                        "mm_processor_kwargs": dict(mm_kwargs),
                        "messages": messages,
                    },
                )

            outputs = self.vlm_server.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"video": image_data},
                    "mm_processor_kwargs": mm_kwargs,
                },
                SamplingParams(max_tokens=2048, temperature=0.01),
                use_tqdm=False,
            )

            result = outputs[0].outputs[0].text.strip()
            if vlm_out:
                refiner_debug.write_text(vlm_out, "vlm_raw_output.txt", result)
            results.append(result)
        
        return results

    def run_refinement_pipeline(self, trace_steps: list, trace_answer: str = None, max_iterations: int = 1):
        print("\n" + "=" * 70)
        print("Starting Trace Refinement Pipeline")
        print("=" * 70 + "\n")

        trace_answer = (trace_answer or self._extract_trace_answer(trace_steps) or "").strip()
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
        verifier_passed = False

        debug_resolved = None
        if self.refinement_debug_root:
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
            debug_resolved = self._refinement_debug_session_base
        else:
            self._refinement_debug_session_base = None

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
                iteration=iteration,
                history=iteration_history,
                max_iterations=max_iterations,
            )
            final_verifier_raw, final_verifier_output = verifier_raw, verifier_output
            print(f"\n[Verifier Output]\n{verifier_raw}\n")

            if isinstance(verifier_output, dict) and verifier_output.get("verdict") == "PASS":
                print("[Verifier] PASS — stopping refinement loop.")
                verifier_passed = True
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

        if not verifier_passed and all_iterations and len(all_iterations) >= max_iterations:
            print("[Verifier] Final post-refinement diagnosis...")
            final_verifier_raw, final_verifier_output = self._call_verifier(
                current_trace,
                current_answer,
                iteration=max_iterations,
                history=iteration_history,
                max_iterations=max_iterations,
            )
            print(f"\n[Final Verifier Output]\n{final_verifier_raw}\n")

        self._refinement_debug_iter_dir = None

        return {
            "question": self.question,
            "options": self.options,
            "video_path": self.video_path,
            "initial_trace": {"steps": initial_trace},
            "initial_answer": initial_answer,
            "final_trace": {"steps": current_trace},
            "final_answer": current_answer,
            "verifier_raw": final_verifier_raw,
            "verifier_output": final_verifier_output,
            "iteration_history": iteration_history,
            "all_iterations": all_iterations,
            "max_iterations": max_iterations,
            "refinement_debug_root": debug_resolved,
        }


def main():
    VIDEO_PATH = "/share/users/ghazi/cot/VideoDeepResearch/videos/-4PUD-TNhU4.mp4"
    QUESTION = (
        "How many players are below the referee in the frame in the initial faceoff?"
    )
    OPTIONS = [
        "A. 6",
        "B. 2",
        "C. 4",
        "D. 3",
        "E. 8",
    ]
    INITIAL_TRACE_STEPS = [
        "I located the start of the first faceoff at 00:04.",
        "I identified the referee by his striped shirt.",
        "I counted the players from both teams visible below the referee in the frame.",
        "There are 6 players visible below the referee.",
        "Therefore, the correct answer is A. 6.",
    ]
    MAX_ITERATIONS = 1

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script to point to a valid video file.")
        return

    demo = VideoQADemo(
        video_path=VIDEO_PATH,
        question=QUESTION,
        options=OPTIONS,
        dataset_folder="./data",
        use_subtitle=False,
        refinement_debug_root="./debug",
        dense_frame_fps=1.0,
        use_clip_retrieval=False,
        dense_segment_half_width=0.5,
        retrieval_top_k=10,
        dense_frame_embed_batch=8,
        vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        planner_model_name="gpt-5",
        planner_api_base=["https://api.openai.com/v1"],
        planner_api_keys=[os.environ["OPENAI_API_KEY"]],
        chart_mode="vlm",
        chart_model_name="Qwen/Qwen3-VL-8B-Instruct",
    )

    result = demo.run_refinement_pipeline(INITIAL_TRACE_STEPS, max_iterations=MAX_ITERATIONS)

    output_path = "refiner_demo_result.json"
    record = {
        "video_path": VIDEO_PATH,
        "question": QUESTION,
        "options": OPTIONS,
        "initial_trace_steps": INITIAL_TRACE_STEPS,
        "initial_trace_answer": result["initial_answer"],
        "final_trace_steps": result["final_trace"]["steps"],
        "final_answer": result["final_answer"],
        "verifier_raw": result["verifier_raw"],
        "verifier_output": result["verifier_output"],
        "iteration_history": result["iteration_history"],
        "all_iterations": result["all_iterations"],
        "max_iterations": result["max_iterations"],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Question: {QUESTION}")
    verifier_verdict = None if not isinstance(result["verifier_output"], dict) else result["verifier_output"].get("verdict")
    n_iters = len(result.get("all_iterations") or [])
    n_tools = sum(len(it.get("executed_tools") or []) for it in (result.get("all_iterations") or []))
    print(f"Verifier Verdict: {verifier_verdict}")
    print(f"Refinement iterations run: {n_iters}")
    print(f"Total executed tool calls: {n_tools}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
