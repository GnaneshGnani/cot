import os
import sys
import json
import re
import time
import argparse
import traceback
import torch
from pathlib import Path
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from openai import OpenAI
import random
from tqdm import tqdm

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 导入必要的工具函数
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
        print(f'读取文件时出错 {file_path}: {e}')
        return None

def list_to_sha256(lst):
    json_str = json.dumps(lst, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


MAX_DS_ROUND = 20  # 最大对话轮数

class VideoQADemo:    
    def __init__(self, 
                 video_path: str = None,
                 question: str = None,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 10,
                 use_subtitle: bool = True,
                 vlm_model_name: str = None,
                 planner_model_name: str = None,
                 temporal_model_name: str = None,
                 vlm_batch_size: int = 4):
        """
        Demo
        
        Args:
            video_path: 视频文件路径
            question: 问题文本
            answer: 正确答案（用于评估）
            options: 选项列表（可选）
            dataset_folder: 数据集文件夹
            clip_duration: 视频片段时长（秒）
            use_subtitle: 是否使用字幕
            vlm_model_name: VLM模型名称
            planner_model_name: 规划模型名称
            temporal_model_name: 时序定位模型名称
        """
        self.video_path = None
        self.question = None
        self.answer = None
        self.options = []
        self.dataset_folder = dataset_folder
        self.clip_duration = clip_duration
        self.use_subtitle = use_subtitle
        self.vlm_batch_size = vlm_batch_size
        
        # 设置环境变量
        self._setup_environment()
        
        # 初始化模型名称
        self.vlm_model_name = vlm_model_name or os.getenv('API_MODEL_NAME_VLM', 'Qwen/Qwen2-VL-7B-Instruct')
        self.planner_model_name = planner_model_name or os.getenv('API_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')
        self.temporal_model_name = temporal_model_name or os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'deepseek-ai/DeepSeek-V3')
        
        # 初始化API配置
        self._setup_api_config()
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化检索器（仅一次）
        self.retriever = self._initialize_retriever()

        self.duration = 0
        self.subtitles = ""
        self.messages = []

        if video_path and question is not None:
            self.load_sample(video_path, question, answer=answer, options=options)

    def load_sample(self, video_path: str, question: str, answer: str = None, options: list = None):
        self.video_path = video_path
        self.question = question
        self.answer = answer
        self.options = options or []
        self.duration = self._get_video_duration()
        self._ensure_video_clip_embeddings()
        self.subtitles = self._extract_subtitles()
        self.messages = []

        print(f"✓ Sample loaded")
        print(f"  Video: {video_path}")
        print(f"  Duration: {self.duration}s")
        print(f"  Question: {question}")
        if self.subtitles:
            print(f"  Subtitles: {len(self.subtitles)} characters")
    
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        torch.backends.cuda.matmul.allow_tf32 = True
    
    def _setup_api_config(self):
        """设置API配置"""
        # 规划模型API配置
        self.planner_api_base = os.getenv('API_BASE_URL', 'http://localhost:8000/v1').split(',')
        self.planner_api_keys = os.getenv('API_KEY', 'EMPTY').split(',')
        
        # 时序定位模型API配置
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
        
        # 创建临时args对象
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
        """确保当前视频的clip embedding已准备好"""
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
        """获取视频时长"""
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            with VideoFileClip(self.video_path) as video:
                return int(video.duration)
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            return 300  # 默认5分钟
    
    def _extract_subtitles(self):
        """提取字幕"""
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
        """构建初始提示"""
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
        """批量处理视频片段（按batch送入vLLM）"""
        if not tasks:
            return []
        all_results = []
        batch_size = max(1, int(self.vlm_batch_size))
        for start in range(0, len(tasks), batch_size):
            all_results.extend(self._process_batch(tasks[start:start + batch_size]))
        return all_results

    def _process_batch(self, tasks: list):
        batch_inputs = []
        valid_indices = []
        results = ["Error: No valid frames"] * len(tasks)

        for idx, (prompt, image_paths, timestamps) in enumerate(tasks):
            image_data = []
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
                    image_data.append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

            if not image_data:
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
            batch_inputs.append(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"video": image_data},
                    "mm_processor_kwargs": {
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": 768 * 28 * 28,
                        "fps": fps,
                    },
                }
            )
            valid_indices.append(idx)

        if not batch_inputs:
            return results

        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = self.vlm_server.generate(
            batch_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for out_idx, output in enumerate(outputs):
            original_idx = valid_indices[out_idx]
            results[original_idx] = output.outputs[0].text.strip()
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
    
    # ==================== 工具处理函数 ====================
    
    def _process_temporal_grounding(self, output_text: str) -> str:
        print("\n[Tool] Temporal Grounding Agent")
        
        pattern = r"<temporal_grounding_agent>([^<]+)</temporal_grounding_agent>"
        try:
            question = re.findall(pattern, output_text)[0]
        except:
            print("Warning: No valid temporal_grounding_agent found")
            return ""
        
        # 构建代理初始提示
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
    
    def run(self):
        """运行完整的多轮工具调用流程"""
        print("\n" + "="*70)
        print("Starting Video QA Demo - Multi-Turn Tool Calling")
        print("="*70 + "\n")
        
        # 构建初始提示
        initial_prompt = self._build_initial_prompt()
        self.messages = [{
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt}]
        }]
        
        cur_turn = 0
        trace_blocks = []
        
        # 多轮对话循环
        while cur_turn < MAX_DS_ROUND:
            print(f"\n{'='*70}")
            print(f"Round {cur_turn + 1}/{MAX_DS_ROUND}")
            print(f"{'='*70}")
            cur_trace = [f"[Round] {cur_turn + 1}/{MAX_DS_ROUND}"]
            
            # 调用规划模型
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
            
            # 记录规划器输出
            self.messages.append({'role': 'assistant', 'content': output_text})
            cur_turn += 1
            
            # 检查是否有最终答案
            if '<answer>' in output_text:
                answer = self._extract_final_answer(output_text)
                print(f"\n{'='*70}")
                print(f"Final Answer: {answer}")
                print(f"{'='*70}\n")
                
                # 评估准确性
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
            
            # 处理工具调用
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
            
            # 检查是否达到最大轮数
            if cur_turn >= MAX_DS_ROUND:
                self.messages.append({
                    'role': 'user',
                    'content': 'Maximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'
                })
                print("\n[System] Maximum rounds reached, forcing answer...")
                cur_trace.append("[System]\nMaximum rounds reached, forcing answer...")

            trace_blocks.append("\n\n".join(cur_trace))
        
        print(f"\n{'='*70}")
        print("Max rounds reached without answer)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="/share/data/drive_1/ghazi/VideoMathQA",
        help="Path to VideoMathQA folder containing annotations and videos/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videomathqa_traces.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    ann_json = os.path.join(args.benchmark_dir, "annotations.json")
    ann_jsonl = os.path.join(args.benchmark_dir, "annotations.jsonl")
    ann_path = ann_json if os.path.exists(ann_json) else ann_jsonl

    if not os.path.exists(ann_path):
        print(f"Error: annotation file not found under {args.benchmark_dir}")
        return

    with open(ann_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    records = []
    shared_demo = VideoQADemo(
        dataset_folder=args.benchmark_dir,
        clip_duration=10,
        use_subtitle=False,
    )

    def _format_item_for_save(item):
        item_out = dict(item)
        steps = item_out.get("steps")
        if isinstance(steps, str):
            try:
                parsed = json.loads(steps)
                if isinstance(parsed, dict):
                    def _step_key(k):
                        try:
                            return int(str(k))
                        except Exception:
                            return str(k)
                    item_out["steps"] = [parsed[k] for k in sorted(parsed.keys(), key=_step_key)]
            except Exception:
                pass
        return item_out

    for i, item in enumerate(data[:50]):
        question = item.get("question", "")
        options = item.get("options", []) or []
        item_out = _format_item_for_save(item)

        gt = item.get("answer", "")
        video_id = item.get("videoID", item.get("video_id", item.get("video", "")))
        video_file = video_id if str(video_id).endswith(".mp4") else f"{video_id}.mp4"
        video_path = os.path.join(args.benchmark_dir, "videos", video_file)

        print(f"\n[{i+1}/{len(data)}] question_id={item.get('question_id')} video={video_file}")

        if not os.path.exists(video_path):
            records.append({
                "question": question,
                "gt": gt,
                "trace": ["[System]\nVideo file not found."],
                "video": video_path,
                "others": item_out,
            })
            continue

        try:
            shared_demo.load_sample(
                video_path=video_path,
                question=question,
                answer=gt,
                options=options,
            )
            result = shared_demo.run()
            trace_lines = [trace_block.splitlines() for trace_block in result.get("round_traces", [])]

            records.append({
                "question": question,
                "gt": gt,
                "trace": trace_lines,
                "video": video_path,
                "pred": result.get("pred_answer", "-"),
                "is_correct": result.get("is_correct", False),
                "others": item_out,
            })
        except Exception as e:
            print(f"Error processing sample {i+1}/{len(data)} (video={video_file}): {e}")
            records.append({
                "question": question,
                "gt": gt,
                "trace": [f"[System]\nError processing sample: {e}", traceback.format_exc()],
                "video": video_path,
                "pred": "-",
                "is_correct": False,
                "others": item_out,
            })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(records)} traces to: {args.output}")

if __name__ == "__main__":
    main()
