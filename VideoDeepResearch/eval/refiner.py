import fcntl
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    VideoFileClip = None

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from prompt import (
    initial_input_template_subtitle,
    initial_input_template_temporal_grounding_agent,
    initial_input_template_temporal_grounding_agent_wo_subtitle,
    initial_input_template_wo_subtitle,
)
from refiner_agents import RefinerAgentsMixin
from refiner_tools import RefinerToolsMixin
from refiner_utils import RefinerUtilsMixin
from retriever_languagebind import Retrieval_Manager
from video_utils import (
    extract_subtitles,
    extract_video_clip,
    parse_subtitle_time,
    robust_eval,
    timestamp_to_clip_path,
)

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


MAX_DS_ROUND = 20


class VideoQADemo(RefinerUtilsMixin, RefinerToolsMixin, RefinerAgentsMixin):
    def __init__(self,
                 video_path: str,
                 question: str,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 5,
                 use_subtitle: bool = True,
                 vlm_model_name: str = None,
                 planner_model_name: str = None,
                 temporal_model_name: str = None):
        self.video_path = video_path
        self.question = question
        self.answer = answer
        self.options = options or []
        self.dataset_folder = dataset_folder
        self.clip_duration = clip_duration
        self.use_subtitle = use_subtitle

        self._setup_environment()

        self.vlm_model_name = vlm_model_name or os.getenv('API_MODEL_NAME_VLM', 'Qwen/Qwen2-VL-7B-Instruct')
        self.planner_model_name = planner_model_name or os.getenv('API_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')
        self.temporal_model_name = temporal_model_name or os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'deepseek-ai/DeepSeek-V3')

        # self.vlm_model_name = vlm_model_name or os.getenv('API_MODEL_NAME_VLM', 'Qwen/Qwen2.5-VL-7B-Instruct')
        # self.planner_model_name = planner_model_name or os.getenv('API_MODEL_NAME', 'Qwen/Qwen2.5-VL-7B-Instruct')
        # self.temporal_model_name = temporal_model_name or os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'Qwen/Qwen2.5-VL-7B-Instruct')

        self._setup_api_config()

        self._initialize_models()

        self.duration = self._get_video_duration()

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
    
    def _setup_api_config(self):
        self.planner_api_base = os.getenv('API_BASE_URL', 'http://localhost:8000/v1').split(',')
        self.planner_api_keys = os.getenv('API_KEY', 'EMPTY').split(',')

        self.temporal_api_base = os.getenv('API_BASE_URL_TEMPORAL_GROUNDING', 'http://localhost:8001/v1').split(',')
        self.temporal_api_keys = os.getenv('API_KEY_TEMPORAL_GROUNDING', 'EMPTY').split(',')
    
    def _initialize_models(self):
        print("Initializing VLM model...")
        
        _mm_kw = {
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        }
        self.vlm_server = LLM(
            model=self.vlm_model_name,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32768,
            enable_chunked_prefill=True,
            enforce_eager=True,
            mm_processor_kwargs=_mm_kw,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name, 
            use_fast=True
        )
        self.processor.tokenizer.padding_side = 'left'
        
        print(f"✓ VLM model loaded: {self.vlm_model_name}")
    
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
    
    def _build_initial_prompt(self):
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

    def _text2text(self, message: list, model_name: str, api_base: list, api_keys: list, queue_type: str = 'planner') -> str:
        print("\n" + "=" * 70)
        print("message: ", message)
        print("model_name: ", model_name)
        print("api_base: ", api_base)
        print("api_keys: ", api_keys)
        print("queue_type: ", queue_type)
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
        results = []
        
        for prompt, image_paths, timestamps in tasks:
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

    def _process_temporal_grounding(self, output_text: str) -> str:
        print("\n[Tool] Temporal Grounding Agent")
        
        pattern = r"<temporal_grounding_agent>([^<]+)</temporal_grounding_agent>"
        try:
            question = re.findall(pattern, output_text)[0]
        except:
            print("Warning: No valid temporal_grounding_agent found")
            return ""

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

    def run_refinement_pipeline(self, trace_steps: list, trace_answer: str = None, max_iterations: int = 3):
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

        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")

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
        }
    
    def run(self):
        print("\n" + "="*70)
        print("Starting Video QA Demo - Multi-Turn Tool Calling")
        print("="*70 + "\n")

        initial_prompt = self._build_initial_prompt()
        self.messages = [{
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt}]
        }]
        
        cur_turn = 0
        trace_blocks = []

        while cur_turn < MAX_DS_ROUND:
            print(f"\n{'='*70}")
            print(f"Round {cur_turn + 1}/{MAX_DS_ROUND}")
            print(f"{'='*70}")
            cur_trace = [f"[Round] {cur_turn + 1}/{MAX_DS_ROUND}"]

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

            self.messages.append({'role': 'assistant', 'content': output_text})
            cur_turn += 1

            if '<answer>' in output_text:
                answer = self._extract_final_answer(output_text)
                print(f"\n{'='*70}")
                print(f"Final Answer: {answer}")
                print(f"{'='*70}\n")

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
    VIDEO_PATH = "/fs/nexus-scratch/gnanesh/cot/VideoMathQA/videos/875b24c9-a2ab-4965-8186-76495a5b553d.mp4"
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
        clip_duration=5,
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
