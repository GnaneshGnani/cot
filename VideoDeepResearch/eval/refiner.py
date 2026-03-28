import os
import sys
import json
import re
import time
import torch
from pathlib import Path
from transformers import AutoProcessor
from vllm import LLM
from PIL import Image
from openai import OpenAI
import random

# Add parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import required utility functions
from video_utils import (
    timestamp_to_clip_path, extract_subtitles, parse_subtitle_time,
    timestamp_to_frames, extract_video_clip, robust_eval
)
from retriever_languagebind import Retrieval_Manager
from prompt import (
    initial_input_template_subtitle, 
    initial_input_template_wo_subtitle,
    initial_input_template_temporal_grounding_agent,
    initial_input_template_temporal_grounding_agent_wo_subtitle
)
from refine_prompt import verifier_propmt, planner_prompt

import os
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp
import hashlib
import json
import os
import time
import pickle
import fcntl
from pathlib import Path

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


MAX_DS_ROUND = 20  # Maximum conversation rounds

class VideoQADemo:    
    def __init__(self, 
                 video_path: str,
                 question: str,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 10,
                 use_subtitle: bool = True,
                 vlm_model_name: str = None,
                 planner_model_name: str = None,
                 temporal_model_name: str = None):
        """
        Demo
        
        Args:
            video_path: 
            question: Question text
            answer: Ground-truth answer for evaluation
            options: Optional answer choices
            dataset_folder: Dataset folder
            clip_duration: Video clip duration in seconds
            use_subtitle: Whether to use subtitles
            vlm_model_name: VLM model name
            planner_model_name: Planner model name
            temporal_model_name: Temporal grounding model name
        """
        self.video_path = video_path
        self.question = question
        self.answer = answer
        self.options = options or []
        self.dataset_folder = dataset_folder
        self.clip_duration = clip_duration
        self.use_subtitle = use_subtitle
        
        # Set environment variables
        self._setup_environment()
        
        # Initialize model names
        self.vlm_model_name = vlm_model_name or os.getenv('API_MODEL_NAME_VLM', 'Qwen/Qwen2-VL-7B-Instruct')
        self.planner_model_name = planner_model_name or os.getenv('API_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')
        self.temporal_model_name = temporal_model_name or os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'deepseek-ai/DeepSeek-V3')
        
        # Initialize API configuration
        self._setup_api_config()
        
        # Initialize models
        self._initialize_models()
        
        # Get video duration
        self.duration = self._get_video_duration()
        
        # Initialize retriever
        self.retriever = self._initialize_retriever()
        
        # If preprocessing was not done in advance, compute clip embeddings for this video
        self._ensure_video_clip_embeddings()
        
        # Extract subtitles
        self.subtitles = self._extract_subtitles()
        
        # Conversation history
        self.messages = []
        
        print(f"✓ Demo initialized successfully")
        print(f"  Video: {video_path}")
        print(f"  Duration: {self.duration}s")
        print(f"  Question: {question}")
        if self.subtitles:
            print(f"  Subtitles: {len(self.subtitles)} characters")
    
    def _setup_environment(self):
        """Set environment variables."""
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        torch.backends.cuda.matmul.allow_tf32 = True
    
    def _setup_api_config(self):
        """Set API configuration."""
        # Planner model API configuration
        self.planner_api_base = os.getenv('API_BASE_URL', 'http://localhost:8000/v1').split(',')
        self.planner_api_keys = os.getenv('API_KEY', 'EMPTY').split(',')
        
        # Temporal grounding model API configuration
        self.temporal_api_base = os.getenv('API_BASE_URL_TEMPORAL_GROUNDING', 'http://localhost:8001/v1').split(',')
        self.temporal_api_keys = os.getenv('API_KEY_TEMPORAL_GROUNDING', 'EMPTY').split(',')
    
    def _initialize_models(self):
        print("Initializing VLM model...")
        
        self.vlm_server = LLM(
            model=self.vlm_model_name,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32768,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name, 
            use_fast=True
        )
        self.processor.tokenizer.padding_side = 'left'
        
        print(f"✓ VLM model loaded: {self.vlm_model_name}")
    
    def _initialize_retriever(self):
        print("Initializing retriever...")
        
        # Create a temporary args object
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
        """Ensure clip embeddings for the current video are ready."""
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
        """Get video duration."""
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            with VideoFileClip(self.video_path) as video:
                return int(video.duration)
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            return 300  # Default to 5 minutes
    
    def _extract_subtitles(self):
        """Extract subtitles."""
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
    
    def _build_initial_prompt(self):
        """Build the initial prompt."""
        question_text = self.question.strip()
        
        if self.use_subtitle:
            prompt = initial_input_template_subtitle.format(
                question=question_text,
                duration=self.duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=MAX_DS_ROUND
            )
        else:
            prompt = initial_input_template_wo_subtitle.format(
                question=question_text,
                duration=self.duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=MAX_DS_ROUND
            )
        
        return prompt.replace('thinking>', 'think>')
    
    def _text2text(self, message: list, model_name: str, api_base: list, api_keys: list, queue_type: str = 'planner') -> str:
        
        folder_path = '_temporal' if queue_type == 'temporal' else '_planner'
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
        """Process video clips in batch."""
        results = []
        
        for prompt, image_paths, timestamps in tasks:
            # Load images
            image_data = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path)
                        image.verify()
                        image = Image.open(img_path)
                        
                        # Resize if needed
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
            
            # Build messages
            content = [
                {"type": "video", "video": image_paths},
                {"type": "text", "text": prompt}
            ]
            messages = [{"role": "user", "content": content}]
            
            # Format prompt
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate response
            fps = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 2.0
            
            outputs = self.vlm_server.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"video": image_data},
                    "mm_processor_kwargs": {
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": 768 * 28 * 28,
                        "fps": fps,
                    },
                },
                use_tqdm=False
            )
            
            result = outputs[0].outputs[0].text.strip()
            results.append(result)
        
        return results
    
    def _extract_final_answer(self, text: str) -> str:
        try:
            answer_content = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)[-1].strip()
            answer_content = re.sub(r'\s+', ' ', answer_content)
            return answer_content if answer_content else '-'
        except:
            return '-'

    def _normalize_answer(self, answer: str) -> str:
        return re.sub(r'\s+', ' ', str(answer)).strip().lower()

    def _format_question_with_options(self) -> str:
        if not self.options:
            return self.question.strip()
        return self.question.strip() + "\nOptions:\n" + "\n".join(self.options)

    def _format_trace_steps(self, trace_steps: list) -> str:
        return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(trace_steps))

    def _extract_trace_answer(self, trace_steps: list) -> str:
        for step in reversed(trace_steps):
            match = re.search(r"final answer\s*:\s*(.*)", str(step), flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    # ==================== Refiner tool helpers ====================

    def _safe_float(self, value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_json_payload(self, text):
        if isinstance(text, (dict, list)):
            return text
        if not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        candidates.append(text)

        for left, right in (("{", "}"), ("[", "]")):
            start = text.find(left)
            end = text.rfind(right)
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start:end + 1])

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except Exception:
                parsed = robust_eval(candidate)
                if isinstance(parsed, (dict, list)):
                    return parsed
        return None

    def _get_refine_tool_calls(self, output_text: str, tool_name: str) -> list:
        payload = self._extract_json_payload(output_text)
        if not isinstance(payload, dict):
            return []

        tool_calls = payload.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return []

        matches = []
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("tool") != tool_name:
                continue
            arguments = call.get("arguments", {})
            matches.append(arguments if isinstance(arguments, dict) else {})
        return matches

    def _format_refine_tool_result(self, tool_name: str, arguments: dict, result) -> str:
        return (
            f"The tool results for {tool_name}({json.dumps(arguments, ensure_ascii=False)}) are:\n"
            f"{json.dumps(result, ensure_ascii=False)}\n"
        )

    def _get_time_range(self, start_time=None, end_time=None):
        start = self._safe_float(start_time, 0.0)
        end = self._safe_float(end_time, float(self.duration))
        start = max(0.0, min(start, float(self.duration)))
        end = max(0.0, min(end, float(self.duration)))
        if end <= start:
            end = min(float(self.duration), start + max(1.0, float(self.clip_duration)))
        return start, end

    def _get_frames_for_range(self, start_time=None, end_time=None, fps: float = 2.0):
        start, end = self._get_time_range(start_time, end_time)
        frame_paths, timestamps = timestamp_to_clip_path(
            self.dataset_folder, start, end, self.video_path, fps=fps
        )
        return frame_paths, timestamps, start, end

    def _get_frame_at_timestamp(self, timestamp: float):
        timestamp = self._safe_float(timestamp, 0.0)
        frame_paths, timestamps, _, _ = self._get_frames_for_range(timestamp, timestamp, fps=1.0)
        if not frame_paths:
            return None, None
        best_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
        return frame_paths[best_idx], float(timestamps[best_idx])

    def _run_vlm_json(self, prompt: str, frame_paths: list, timestamps: list, default_result):
        if not frame_paths:
            return default_result

        output_text = self._batch_video2text([(prompt, frame_paths, timestamps)])[0]
        parsed = self._extract_json_payload(output_text)
        if parsed is not None:
            return parsed

        if isinstance(default_result, dict):
            result = dict(default_result)
            result["raw_output"] = output_text
            return result
        return default_result

    def _get_asr_result_from_subtitles(self, start_time=None, end_time=None):
        try:
            subtitle_segments = extract_subtitles(self.video_path)
        except Exception:
            subtitle_segments = []

        if start_time is None and end_time is None:
            filtered = subtitle_segments
        else:
            start, end = self._get_time_range(start_time, end_time)
            filtered = [x for x in subtitle_segments if x[1] >= start and x[0] <= end]

        segments = [
            {
                "start": float(seg[0]),
                "end": float(seg[1]),
                "text": seg[2],
                "speaker": None,
                "confidence": 1.0,
            }
            for seg in filtered
        ]
        transcript = " ".join(seg["text"] for seg in segments).strip()
        return {
            "language_detected": "unknown",
            "transcript": transcript,
            "full_transcript": transcript,
            "segments": segments,
            "words": [],
        }

    def _get_preprocessed_artifacts(self) -> dict:
        asr_result = self._get_asr_result_from_subtitles()
        return {
            "asr_transcript": asr_result.get("full_transcript", ""),
            "dense_captions": None,
            "audio_events": None,
            "keyframe_index": [],
        }

    def _build_verifier_prompt(self, trace_steps: list, trace_answer: str) -> str:
        return (
            verifier_propmt.strip()
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nTRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nANSWER:\n"
            + trace_answer
            + "\n\nVIDEO:\n"
            + self.video_path
        )

    def _build_planner_prompt(self, trace_steps: list, trace_answer: str, diagnosis) -> str:
        diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, indent=2) if isinstance(diagnosis, dict) else str(diagnosis)
        artifacts_text = json.dumps(self._get_preprocessed_artifacts(), ensure_ascii=False, indent=2)
        return (
            planner_prompt.strip()
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nTRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nANSWER:\n"
            + trace_answer
            + "\n\nDIAGNOSIS:\n"
            + diagnosis_text
            + "\n\nPREPROCESSED_ARTIFACTS:\n"
            + artifacts_text
        )

    def _call_verifier(self, trace_steps: list, trace_answer: str):
        prompt = self._build_verifier_prompt(trace_steps, trace_answer)
        messages = [{"role": "user", "content": prompt}]
        raw_output = self._text2text(
            messages, self.planner_model_name, self.planner_api_base, self.planner_api_keys
        )
        parsed_output = self._extract_json_payload(raw_output)
        return raw_output, parsed_output if isinstance(parsed_output, dict) else None

    def _call_planner(self, trace_steps: list, trace_answer: str, diagnosis):
        prompt = self._build_planner_prompt(trace_steps, trace_answer, diagnosis)
        messages = [{"role": "user", "content": prompt}]
        raw_output = self._text2text(
            messages, self.planner_model_name, self.planner_api_base, self.planner_api_keys
        )
        parsed_output = self._extract_json_payload(raw_output)
        return raw_output, parsed_output if isinstance(parsed_output, dict) else None

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
            "video_qa_reanswerer": self._process_video_qa_reanswerer,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}\n"

        payload = json.dumps({
            "tool_calls": [{"tool": tool_name, "arguments": arguments or {}}]
        }, ensure_ascii=False)
        return handler(payload)

    def _execute_refine_plan(self, planner_plan: dict) -> list:
        if not isinstance(planner_plan, dict):
            return []

        tool_calls = planner_plan.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return []

        ordered_calls = sorted(
            [call for call in tool_calls if isinstance(call, dict)],
            key=lambda call: int(call.get("step", 0) or 0)
        )

        execution_results = []
        for call in ordered_calls:
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            output = self._execute_refine_tool_call(tool_name, arguments if isinstance(arguments, dict) else {})
            execution_results.append({
                "step": call.get("step"),
                "tool": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "purpose": call.get("purpose", ""),
                "depends_on": call.get("depends_on", []),
                "output": output.strip(),
            })
        return execution_results

    def _process_temporal_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "temporal_grounder")
        if not calls:
            return ""

        print("\n[Tool] Temporal Grounder")
        results = []
        topk = int(os.getenv('TOPK', '5'))

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            segments = []
            if query:
                try:
                    clip_results = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, top_k=topk, total_duration=self.duration
                    )
                    for clip_path, score in clip_results:
                        clip_number = int(os.path.basename(clip_path).split('_')[1])
                        start = float(clip_number * self.clip_duration)
                        end = float(min(self.duration, start + self.clip_duration))
                        segments.append({
                            "start": start,
                            "end": end,
                            "confidence": float(score),
                        })
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
                        frames.append({
                            "frame_path": frame_path,
                            "timestamp": float(frame_ts),
                        })
            elif query:
                mode = "query"
                try:
                    clip_results = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, top_k=num_frames, total_duration=self.duration
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    clip_results = []

                for clip_path, score in clip_results[:num_frames]:
                    clip_number = int(os.path.basename(clip_path).split('_')[1])
                    ts = clip_number * self.clip_duration + self.clip_duration / 2
                    frame_path, frame_ts = self._get_frame_at_timestamp(ts)
                    if frame_path:
                        frames.append({
                            "frame_path": frame_path,
                            "timestamp": float(frame_ts),
                            "relevance_score": float(score),
                        })

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
            result = self._get_asr_result_from_subtitles(
                arguments.get("start_time"), arguments.get("end_time")
            )
            results.append(self._format_refine_tool_result("asr", arguments, result))

        return "".join(results)

    def _process_audio_grounder(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "audio_grounder")
        if not calls:
            return ""

        print("\n[Tool] Audio Grounder")
        results = []

        for arguments in calls:
            asr_result = self._get_asr_result_from_subtitles(
                arguments.get("start_time"), arguments.get("end_time")
            )
            summary = (
                "Speech subtitles are available in this range, but non-speech audio grounding is unavailable."
                if asr_result["segments"] else
                "Non-speech audio grounding is unavailable in this runner."
            )
            result = {
                "query": str(arguments.get("query", "")).strip(),
                "events": [],
                "audio_summary": summary,
            }
            results.append(self._format_refine_tool_result("audio_grounder", arguments, result))

        return "".join(results)

    def _process_ocr(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "ocr")
        if not calls:
            return ""

        print("\n[Tool] OCR")
        results = []

        for arguments in calls:
            frame_path = arguments.get("frame_path")
            frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)

            if not frame_path:
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            source = frame_path if frame_path else f"{self.video_path}@{frame_ts}"
            result = {"source": source, "detections": [], "full_text": ""}

            if frame_path and os.path.exists(frame_path):
                try:
                    import pytesseract

                    data = pytesseract.image_to_data(
                        Image.open(frame_path), output_type=pytesseract.Output.DICT
                    )
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
                        detections.append({
                            "text": text,
                            "bbox": [x, y, x + w, y + h],
                            "confidence": conf / 100.0 if conf > 1 else conf,
                            "text_type": "scene_text",
                        })
                    result = {
                        "source": source,
                        "detections": detections,
                        "full_text": "\n".join(x["text"] for x in detections),
                    }
                except Exception:
                    prompt = (
                        'Extract all visible text and return JSON: '
                        '{"source":"","detections":[{"text":"","bbox":[0,0,0,0],"confidence":0.0,"text_type":"scene_text"}],"full_text":""}.'
                    )
                    result = self._run_vlm_json(
                        prompt,
                        [frame_path],
                        [float(frame_ts)],
                        result,
                    )
                    if isinstance(result, dict):
                        result.setdefault("source", source)

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
            if not frame_path:
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            default_result = {
                "query": query,
                "detections": [],
                "spatial_description": "",
            }
            prompt = (
                'Detect objects matching the query and return JSON: '
                '{"query":"","detections":[{"label":"","bbox":[0,0,0,0],"confidence":0.0,"mask_path":null,"area_fraction":0.0}],'
                '"spatial_description":""}. '
                f"Query: {query}"
            )
            result = self._run_vlm_json(
                prompt,
                [frame_path] if frame_path else [],
                [float(frame_ts)],
                default_result,
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
            if not frame_path:
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            default_result = {
                "query": query,
                "count": 0,
                "confidence": 0.0,
                "detections": [],
                "notes": "",
            }
            prompt = (
                'Count the queried objects and return JSON: '
                '{"query":"","count":0,"confidence":0.0,"detections":[{"bbox":[0,0,0,0],"instance_confidence":0.0}],"notes":""}. '
                f"Query: {query}"
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
                'Return JSON: {"video_duration":0.0,"captioned_range":{"start":0.0,"end":0.0},'
                '"captions":[{"start":0.0,"end":0.0,"visual":"","audio":"","on_screen_text":"","actions":[],"objects":[]}],'
                '"overall_summary":""}. '
                f"Granularity: {granularity}. Focus query: {focus_query}"
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
                arguments.get("start_time"), arguments.get("end_time"), fps=2.0
            )
            query = str(arguments.get("query", "")).strip()
            default_result = {
                "analyzed_range": {"start": start, "end": end},
                "actions": [],
                "query_response": None,
            }
            prompt = (
                'Return JSON: {"analyzed_range":{"start":0.0,"end":0.0},"actions":[{"action":"","start":0.0,"end":0.0,'
                '"confidence":0.0,"actor":""}],"query_response":null}. '
                f"Focus on human actions. Query: {query}"
            )
            result = self._run_vlm_json(prompt, frame_paths, timestamps, default_result)
            results.append(self._format_refine_tool_result("action_recognizer", arguments, result))

        return "".join(results)

    def _process_video_qa_reanswerer(self, output_text: str) -> str:
        calls = self._get_refine_tool_calls(output_text, "video_qa_reanswerer")
        if not calls:
            return ""

        print("\n[Tool] Video QA Re-answerer")
        results = []

        for arguments in calls:
            question = str(arguments.get("question", "")).strip()
            frame_paths, timestamps, _, _ = self._get_frames_for_range(None, None, fps=2.0)
            default_result = {
                "question": question,
                "answer": "",
                "reasoning": "",
                "confidence": 0.0,
                "key_evidence": [],
            }
            prompt = (
                'Answer the question from the video and return JSON: '
                '{"question":"","answer":"","reasoning":"","confidence":0.0,'
                '"key_evidence":[{"timestamp":0.0,"modality":"visual","observation":""}]}. '
                f"Question: {question}"
            )
            result = self._run_vlm_json(prompt, frame_paths, timestamps, default_result)
            results.append(self._format_refine_tool_result("video_qa_reanswerer", arguments, result))

        return "".join(results)

    def _process_refine_tool_calls(self, output_text: str) -> str:
        tool_result = ""
        tool_result += self._process_temporal_grounder(output_text)
        tool_result += self._process_frame_retriever(output_text)
        tool_result += self._process_asr(output_text)
        tool_result += self._process_audio_grounder(output_text)
        tool_result += self._process_ocr(output_text)
        tool_result += self._process_spatial_grounder(output_text)
        tool_result += self._process_counter(output_text)
        tool_result += self._process_dense_captioner(output_text)
        tool_result += self._process_action_recognizer(output_text)
        tool_result += self._process_video_qa_reanswerer(output_text)
        return tool_result
    
    # ==================== Tool processing functions ====================
    
    def _process_temporal_grounding(self, output_text: str) -> str:
        print("\n[Tool] Temporal Grounding Agent")
        
        pattern = r"<temporal_grounding_agent>([^<]+)</temporal_grounding_agent>"
        try:
            question = re.findall(pattern, output_text)[0]
        except:
            print("Warning: No valid temporal_grounding_agent found")
            return ""
        
        # Build the agent's initial prompt
        if self.use_subtitle:
            agent_prompt = initial_input_template_temporal_grounding_agent.format(
                clip_duration=10, question=question, duration=self.duration
            )
        else:
            agent_prompt = initial_input_template_temporal_grounding_agent_wo_subtitle.format(
                clip_duration=10, question=question, duration=self.duration
            )
        
        agent_prompt = agent_prompt.replace('thinking>', 'think>')
        agent_messages = [{"role": "user", "content": agent_prompt}]
        
        tool_call = self._text2text(agent_messages, self.temporal_model_name, 
                                    self.temporal_api_base, self.temporal_api_keys, queue_type='temporal')
        agent_messages.append({"role": "assistant", "content": tool_call})
        
        tool_results = self._process_tool_calls_for_temporal(tool_call)
        agent_messages.append({
            "role": "user",
            "content": tool_results + "\nNow you should call the video reader to check the video segments."
        })
        
        tool_call = self._text2text(agent_messages, self.temporal_model_name,
                                    self.temporal_api_base, self.temporal_api_keys, queue_type='temporal')
        agent_messages.append({"role": "assistant", "content": tool_call})
        
        tool_results = self._process_tool_calls_for_temporal(tool_call)
        agent_messages.append({
            "role": "user",
            "content": tool_results + "\nNow you should output the final video segments."
        })
        
        tool_call = self._text2text(agent_messages, self.temporal_model_name,
                                    self.temporal_api_base, self.temporal_api_keys, queue_type='temporal')
        
        if '<answer>' in tool_call:
            answer_text = re.findall(r'<answer>([^<]+)</answer>', tool_call, re.DOTALL)[-1].strip()
            b_idx, e_idx = answer_text.find('['), answer_text.find(']')
            if b_idx != -1 and e_idx != -1:
                answer_text = answer_text[b_idx:e_idx+1]
                intervals = robust_eval(answer_text)
                result = f'There are {len(intervals)} related segments in the video: {intervals}'
                print(f"  Found {len(intervals)} segments")
                return result
        
        return "No segments found"
    
    def _process_video_reader(self, output_text: str) -> str:
        print("\n[Tool] Video Reader")
        
        pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
        matches = re.findall(pattern, output_text.strip())
        
        if not matches:
            return ""
        
        tasks = []
        time_matches = [match[0] for match in matches]
        question_matches = [match[1] for match in matches]
        
        for query, time_match in zip(question_matches, time_matches):
            begin_time, end_time = time_match.split(':')
            begin_time, end_time = float(begin_time), float(end_time)
            
            video_clip, timestamps = timestamp_to_clip_path(
                self.dataset_folder, begin_time, end_time, 
                self.video_path, fps=2.0
            )
            
            if len(video_clip) > 0:
                query_formatted = (
                    f"Please watch the given video and answer the following question: {query}\n"
                    "Output the detailed video description and the answer in this format: "
                    "The description of the video is:YOUR_DESCRIPTION\nThe answer is:YOUR_ANSWER."
                )
                tasks.append((query_formatted, video_clip, timestamps))
        
        if not tasks:
            return ""
        
        results = self._batch_video2text(tasks)
        
        ans = ""
        for time_match, result in zip(time_matches, results):
            ans += f'The tool result for <video_reader>{time_match}</video_reader> is {result}\n'
            print(f"  Processed segment: {time_match}")
        
        return ans
    
    def _process_video_segment_retriever_text(self, output_text: str) -> str:
        print("\n[Tool] Video Segment Retriever (Text)")
        
        pattern = r"<video_segment_retriever_textual_query>(.*?)</video_segment_retriever_textual_query>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches:
            return ""
        
        results = []
        topk = int(os.getenv('TOPK', '5'))
        
        for match in matches:
            for query in match.split(';'):
                try:
                    video_clip_paths = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, 
                        top_k=topk, total_duration=self.duration
                    )
                    cur_video_paths = [
                        int(video[0].split('/')[-1].split('_')[1]) 
                        for video in video_clip_paths
                    ]
                    results.append(
                        f"The tool results for <video_segment_retriever_textual_query>{query}"
                        f"</video_segment_retriever_textual_query> are:\n{cur_video_paths}\n"
                    )
                    print(f"  Query: {query[:50]}... -> Found {len(cur_video_paths)} clips")
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        return ''.join(results)
    
    def _process_video_segment_retriever_image(self, output_text: str) -> str:
        print("\n[Tool] Video Segment Retriever (Image Query)")
        
        pattern = r"<video_segment_retriever_image_query>(.*?)</video_segment_retriever_image_query>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        pattern = r"<video_segment_retriever_image_query_text>(.*?)</video_segment_retriever_image_query_text>"
        matches_text = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches or not matches_text:
            return ""
        
        results = []
        topk = int(os.getenv('TOPK', '5'))
        
        for match, match_text in zip(matches, matches_text):
            try:
                begin, end = float(match) - 1, float(match) + 1
                query_video_path = extract_video_clip(self.video_path, begin, end)
                
                video_clip_paths = self.retriever.get_informative_clips_with_video_query(
                    match_text, query_video_path,
                    video_path=self.video_path, top_k=topk, total_duration=self.duration
                )
                
                cur_video_paths = []
                for video in video_clip_paths:
                    clip_number = int(video[0].split('/')[-1].split('_')[1])
                    if not clip_number * self.clip_duration <= float(match) <= clip_number * self.clip_duration + self.clip_duration:
                        cur_video_paths.append(clip_number)
                
                results.append(
                    f"The tool results for <video_segment_retriever_image_query>{match}</video_segment_retriever_image_query> are:\n"
                    f"{cur_video_paths}\n"
                )
                print(f"  Query @ {match}: {match_text[:50]}... -> Found {len(cur_video_paths)} clips")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return ''.join(results)
    
    def _process_subtitle_retriever(self, output_text: str) -> str:
        print("\n[Tool] Subtitle Retriever")
        
        pattern = r"<subtitle_retriever>(.*?)</subtitle_retriever>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches:
            return ""
        
        results = []
        topk = 10
        
        for match in matches:
            subtitle_triples = []
            vis = []
            
            for query in match.split(';'):
                try:
                    cur_subtitle_triples = self.retriever.get_informative_subtitles(
                        query, video_path=self.video_path,
                        top_k=topk, total_duration=self.duration
                    )
                    
                    for x in cur_subtitle_triples:
                        if x[0] not in vis:
                            subtitle_triples.append({
                                'begin_timestamp': x[0],
                                'end_timestamp': x[1],
                                'text': x[2]
                            })
                            vis.append(x[0])
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
            
            subtitle_triples = sorted(subtitle_triples, key=lambda x: x['begin_timestamp'])
            results.append(
                f"The tool results for <subtitle_retriever>{match}</subtitle_retriever> are:\n"
                f"{subtitle_triples}\n"
            )
            print(f"  Found {len(subtitle_triples)} subtitle segments")
        
        return ''.join(results)
    
    def _process_subtitle_extractor(self, output_text: str) -> str:
        print("\n[Tool] Subtitle Extractor")
        
        pattern = r"<subtitle_extractor>(.*?)</subtitle_extractor>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches:
            return ""
        
        all_subtitle_triples = extract_subtitles(self.video_path)
        results = []
        for time_match in matches:
            for match in time_match.split(';'):
                try:
                    begin_timestamp = float(match.split(':')[0])
                    end_timestamp = float(match.split(':')[1])
                    cur_subtitle_triples = [
                        {'begin_timestamp': int(x[0]), 'end_timestamp': int(x[1]), 'subtitle': x[2]}
                        for x in all_subtitle_triples if begin_timestamp <= x[0] <= end_timestamp
                    ]
                    results.append(
                        f"The tool results for <subtitle_extractor>{match}</subtitle_extractor> are:\n"
                        f"{cur_subtitle_triples}\n"
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        return ''.join(results)
    
    def _process_video_browser(self, output_text: str) -> str:
        print("\n[Tool] Video Browser")
        
        pattern = r"<video_browser>([^<]+)</video_browser>"
        queries = re.findall(pattern, output_text)
        
        if not queries:
            return ""
        
        query = queries[0]
        video_clip, timestamps = timestamp_to_clip_path(
            self.dataset_folder, 0, self.duration, 
            self.video_path, fps=2.0
        )
        
        ans = self._batch_video2text([(query, video_clip, timestamps)])[0]
        print(f"  Browsed entire video")
        return f"The tool results for <video_browser>{query}</video_browser> is:{ans}\n"
    
    def _process_tool_calls(self, output_text: str) -> str:
        tool_result = ""
        tool_result += self._process_refine_tool_calls(output_text)
        
        if "<temporal_grounding_agent>" in output_text:
            tool_result += self._process_temporal_grounding(output_text)
        
        if "<video_reader>" in output_text:
            tool_result += self._process_video_reader(output_text)
        
        if '<video_segment_retriever_textual_query>' in output_text:
            tool_result += self._process_video_segment_retriever_text(output_text)
        
        if '<video_segment_retriever_image_query>' in output_text:
            tool_result += self._process_video_segment_retriever_image(output_text)
        
        if '<subtitle_retriever>' in output_text:
            tool_result += self._process_subtitle_retriever(output_text)
        
        if '<subtitle_extractor>' in output_text:
            tool_result += self._process_subtitle_extractor(output_text)
        
        if "<video_browser>" in output_text:
            tool_result += self._process_video_browser(output_text)
        
        return tool_result
    
    def _process_tool_calls_for_temporal(self, output_text: str) -> str:
        tool_result = ""
        tool_result += self._process_refine_tool_calls(output_text)
        
        if "<video_reader>" in output_text:
            tool_result += self._process_video_reader(output_text)
        
        if '<video_segment_retriever_textual_query>' in output_text:
            tool_result += self._process_video_segment_retriever_text(output_text)
        
        if '<video_segment_retriever_image_query>' in output_text:
            tool_result += self._process_video_segment_retriever_image(output_text)
        
        if '<subtitle_retriever>' in output_text:
            tool_result += self._process_subtitle_retriever(output_text)
        
        if '<subtitle_extractor>' in output_text:
            tool_result += self._process_subtitle_extractor(output_text)
        
        return tool_result

    def run_refinement_pipeline(self, trace_steps: list, trace_answer: str = None):
        print("\n" + "=" * 70)
        print("Starting Trace Refinement Pipeline")
        print("=" * 70 + "\n")

        trace_answer = (trace_answer or self._extract_trace_answer(trace_steps) or "").strip()

        print("[Verifier] Generating diagnosis...")
        verifier_raw, verifier_output = self._call_verifier(trace_steps, trace_answer)
        print(f"\n[Verifier Output]\n{verifier_raw}\n")

        print("[Planner] Generating plan...")
        planner_raw, planner_output = self._call_planner(
            trace_steps,
            trace_answer,
            verifier_output if verifier_output is not None else verifier_raw,
        )
        print(f"\n[Planner Output]\n{planner_raw}\n")

        print("[Executor] Running planned tool calls...")
        executed_tools = self._execute_refine_plan(planner_output if planner_output is not None else {})
        for item in executed_tools:
            print(f"  Step {item['step']} - {item['tool']}")

        return {
            "question": self.question,
            "options": self.options,
            "video_path": self.video_path,
            "initial_trace": {"steps": trace_steps},
            "initial_answer": trace_answer,
            "verifier_raw": verifier_raw,
            "verifier_output": verifier_output,
            "planner_raw": planner_raw,
            "planner_output": planner_output,
            "executed_tools": executed_tools,
        }
    
    def run(self):
        """Run the full multi-round tool-calling workflow."""
        print("\n" + "="*70)
        print("Starting Video QA Demo - Multi-Turn Tool Calling")
        print("="*70 + "\n")
        
        # Build the initial prompt
        initial_prompt = self._build_initial_prompt()
        self.messages = [{
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt}]
        }]
        
        cur_turn = 0
        trace_blocks = []
        
        # Multi-turn conversation loop
        while cur_turn < MAX_DS_ROUND:
            print(f"\n{'='*70}")
            print(f"Round {cur_turn + 1}/{MAX_DS_ROUND}")
            print(f"{'='*70}")
            cur_trace = [f"[Round] {cur_turn + 1}/{MAX_DS_ROUND}"]
            
            # Call planner model
            print("\n[Planner] Generating response...")
            output_text = self._text2text(
                self.messages, 
                self.planner_model_name,
                self.planner_api_base,
                self.planner_api_keys
            )
            print(f"\n[Planner Output]\n{output_text}\n")
            cur_trace.append(f"[Planner]\n{output_text.strip()}")
            
            if not output_text:
                print("Error: No response from planner")
                cur_trace.append("[System]\nError: No response from planner")
                trace_blocks.append("\n\n".join(cur_trace))
                break
            
            # Record planner output
            self.messages.append({'role': 'assistant', 'content': output_text})
            cur_turn += 1
            
            # Check whether a final answer is present
            if '<answer>' in output_text:
                answer = self._extract_final_answer(output_text)
                print(f"\n{'='*70}")
                print(f"Final Answer: {answer}")
                print(f"{'='*70}\n")
                
                # Evaluate correctness
                is_correct = False
                if self.answer:
                    is_correct = (self._normalize_answer(answer) == self._normalize_answer(self.answer))
                    print(f"Ground Truth: {self.answer}")
                    print(f"Correctness: {'✓ Correct' if is_correct else '✗ Incorrect'}\n")
                trace_blocks.append("\n\n".join(cur_trace))
                
                return {
                    'messages': self.messages,
                    'pred_answer': answer,
                    'ground_truth': self.answer,
                    'is_correct': is_correct,
                    'total_rounds': cur_turn,
                    'round_traces': trace_blocks
                }
            
            # Process tool calls
            print("\n[Tool Processor] Processing tool calls...")
            tool_result = self._process_tool_calls(output_text)
            print(f"\n[Tool Results]\n{tool_result}\n")
            cur_trace.append(f"[TOOL]\n{tool_result.strip()}")
            
            if tool_result:
                self.messages.append({
                    'role': 'user',
                    'content': tool_result + f"\nYou have now engaged in {cur_turn} rounds of conversation, "
                               f"with {MAX_DS_ROUND-cur_turn} calls remaining."
                })
                print(f"\n[System] Tool results provided to planner")
                cur_trace.append("[System]\nTool results provided to planner")
            elif '<answer>' not in output_text:
                self.messages.append({
                    'role': 'user',
                    'content': 'The output is invalid. You should strictly follow the provided xml format!!!'
                })
                print("\n[System] Warning: Invalid output format")
                cur_trace.append("[System]\nWarning: Invalid output format")
            
            # Check whether the maximum number of rounds has been reached
            if cur_turn >= MAX_DS_ROUND:
                self.messages.append({
                    'role': 'user',
                    'content': 'Maximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'
                })
                print("\n[System] Maximum rounds reached, forcing answer...")
                cur_trace.append("[System]\nMaximum rounds reached, forcing answer...")

            trace_blocks.append("\n\n".join(cur_trace))
        
        print(f"\n{'='*70}")
        print("Demo completed (max rounds reached without answer)")
        print(f"{'='*70}\n")
        
        return {
            'messages': self.messages,
            'pred_answer': '-',
            'ground_truth': self.answer,
            'is_correct': False,
            'total_rounds': cur_turn,
            'round_traces': trace_blocks
        }


def main():
    """Main function: run the trace refinement demo."""

    VIDEO_PATH = "/share/data/drive_1/ghazi/VideoMathQA/videos/875b24c9-a2ab-4965-8186-76495a5b553d.mp4"
    QUESTION = (
        "Among Walmart, Target, Whole Foods, and Albertsons, which store shows the highest "
        "discrepancy between customer-rated Store Cleanliness and Value for Dollar, and what "
        "is the approximate magnitude of that difference in percentage points?"
    )
    OPTIONS = [
        "A. Whole Foods, 40%",
        "B. Whole Foods, 65%",
        "C. Walmart, 68%",
        "D. Whole Foods, 69%",
        "E. Walmart, 48%",
    ]
    INITIAL_TRACE_STEPS = [
        "The video investigates why Aldi is considered one of the top value-for-money grocery stores in the U.S. It analyzes Aldi's efficiency-focused design, limited product selection, private label use, and minimalist approach that contribute to high perceived value among customers.",
        "Around the midpoint of the video (~2:30), the focus shifts from Aldis internal strategies to consumer sentiment, emphasizing how customers perceive 'value for dollar' and introducing survey-based satisfaction data.",
        "Two key charts are shown: the first at ~2:45 compares customer satisfaction across grocery chains on 'Store Cleanliness' and 'Availability of Items'; the second at ~3:32 shows Value-for-Dollar ratings from a customer survey.",
        "In Chart 1, Whole Foods' store cleanliness score is 80%.",
        "In Chart 1, Walmart's store cleanliness score is estimated at 30%, based on it falling between the 20% and 40% gridlines.",
        "In Chart 2, Whole Foods' value-for-dollar rating is estimated at 15%, based on it appearing between the 0% and 20% marks.",
        "In Chart 2, Walmarts value-for-dollar rating is estimated at 70%, falling between the 60% and 80% range.",
        "Calculating the discrepancy between cleanliness and value-for-dollar for each store:",
        "Whole Foods: |80 - 15| = 65%; Walmart: |30 - 70| = 40%.",
        "Final answer: Whole Foods has the highest discrepancy between cleanliness and perceived value-for-dollar, at 65%.",
    ]

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script to point to a valid video file.")
        return

    demo = VideoQADemo(
        video_path=VIDEO_PATH,
        question=QUESTION,
        options=OPTIONS,
        dataset_folder="./data",
        clip_duration=10,
        use_subtitle=False,
    )

    result = demo.run_refinement_pipeline(INITIAL_TRACE_STEPS)

    output_path = "refiner_demo_result.json"
    record = {
        "video_path": VIDEO_PATH,
        "question": QUESTION,
        "options": OPTIONS,
        "initial_trace_steps": INITIAL_TRACE_STEPS,
        "initial_trace_answer": result["initial_answer"],
        "verifier_raw": result["verifier_raw"],
        "verifier_output": result["verifier_output"],
        "planner_raw": result["planner_raw"],
        "planner_output": result["planner_output"],
        "executed_tools": result["executed_tools"],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Question: {QUESTION}")
    verifier_verdict = None if not isinstance(result["verifier_output"], dict) else result["verifier_output"].get("verdict")
    planned_calls = 0 if not isinstance(result["planner_output"], dict) else len(result["planner_output"].get("tool_calls", []))
    print(f"Verifier Verdict: {verifier_verdict}")
    print(f"Planned Tool Calls: {planned_calls}")
    print(f"Executed Tool Calls: {len(result['executed_tools'])}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
