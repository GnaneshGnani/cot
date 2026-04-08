import os
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer, LanguageBindVideoTokenizer
import torch
import numpy as np
import cv2
import pickle
import time
import json
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except:
    from moviepy import VideoFileClip, concatenate_videoclips

from decord import VideoReader, cpu
from tqdm import tqdm

import math
import argparse
from video_utils import *
import subprocess
import datetime
import multiprocessing

from FlagEmbedding import BGEM3FlagModel

import argparse

class Retrieval_Manager():
    def __init__(self, args=None, batch_size=1, clip_save_folder=None, clip_duration=30):
        video_model_path = os.getenv("LANGUAGEBIND_VIDEO_MODEL_PATH", "").strip() or "LanguageBind/LanguageBind_Video_FT"
        image_model_path = os.getenv("LANGUAGEBIND_IMAGE_MODEL_PATH", "").strip() or "LanguageBind/LanguageBind_Image"
        video_tokenizer_path = os.getenv("LANGUAGEBIND_VIDEO_TOKENIZER_PATH", "").strip() or video_model_path
        self.video_model_tag = Path(video_model_path.rstrip("/")).name or "video_model"

        clip_type = {
            'video': video_model_path,
            'image': image_model_path,
        }

        self.model = LanguageBind(clip_type=clip_type, cache_dir='./model_cache')

        # Prefer an explicit local model path, then local HF cache snapshots, then repo id.
        bge_model_path = os.getenv("BGE_M3_MODEL_PATH", "").strip()
        if not bge_model_path:
            hf_home = os.getenv("HF_HOME", "/fs/nexus-scratch/gnanesh/.cache/huggingface")
            snapshots_dir = Path(hf_home) / "hub" / "models--BAAI--bge-m3" / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
                if snapshot_dirs:
                    bge_model_path = str(snapshot_dirs[-1])

        if bge_model_path and Path(bge_model_path).exists():
            self.text_retriever = BGEM3FlagModel(bge_model_path, use_fp16=True)
        else:
            self.text_retriever = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        self.model.eval()

        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(
            video_tokenizer_path,
            cache_dir='./model_cache',
        )
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}

        self.clip_embs_cache = {}
        self.frame_embs_cache = {}
        self.dense_frame_embs_cache = {}
        self.batch_size = 1
        self.clip_save_folder = clip_save_folder
        self.args=args

    def _clip_embedding_folder(self):
        return f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}/{self.video_model_tag}'


    def load_model_to_device(self, device):

        self.model.to(device)

        def recursive_to(module):
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.nn.Module):
                    attr.to(device)
                    recursive_to(attr)
                elif isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))
                elif isinstance(attr, (list, tuple)):
                    new_attrs = []
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            item.to(device)
                            recursive_to(item)
                        elif isinstance(item, torch.Tensor):
                            item = item.to(device)
                        new_attrs.append(item)
                    setattr(module, name, type(attr)(new_attrs))

        recursive_to(self.model)

    def load_model_to_cpu(self):
        self.device=torch.device('cpu')
        self.load_model_to_device(torch.device('cpu'))
    
    def load_model_to_gpu(self, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.load_model_to_device(torch.device(f'cuda:{gpu_id}'))

  
    def save_clip(self, clip, clip_save_folder, clip_index, start_time, end_time, fps):
        start_time_str = self.format_time(start_time)
        end_time_str = self.format_time(end_time)
        os.makedirs(clip_save_folder,exist_ok=True)
        clip_path = os.path.join(clip_save_folder, f"clip_{clip_index}_{start_time_str}_to_{end_time_str}.mp4")
        height, width, _ = clip[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        for frame in clip:
            out.write(frame)

        out.release()
        return clip_path

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"

    def parse_time(self, time_str):
        hours, mins, secs = map(int, time_str.split('-'))
        total_seconds = hours * 3600 + mins * 60 + secs
        return total_seconds



    def cut_video(self, video_path, clip_save_folder=None, total_duration=-1):
        valid_clip_paths = set()

        if 'video_haystack_' in video_path:
            duration = VideoFileClip('./benchmark/NIAH-Video/videos/video_haystack.mkv').duration
        else:
            duration = VideoFileClip(video_path).duration
        chunk_number = math.ceil(duration/self.args.clip_duration)

        if os.path.exists(clip_save_folder):
            total_video_clip_paths = []
            for i in range(chunk_number):
                start_time = self.args.clip_duration * i
                end_time = start_time + self.args.clip_duration
                output_filename = f'clip_{i}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4'  
                total_video_clip_paths.append(clip_save_folder+'/'+output_filename)     

            valid_clip_num = 0
            path_li = os.listdir(clip_save_folder)
            for clip_name in path_li:
                try:
                    VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                    valid_clip_paths.add(clip_save_folder+'/'+clip_name)
                    valid_clip_num+=1
                    del total_video_clip_paths[total_video_clip_paths.index(clip_save_folder+'/'+clip_name)]
                except Exception as e: 
                    os.system('rm -rf '+clip_save_folder+'/'+clip_name)
                    
            if valid_clip_num >= chunk_number-3:
                return [], [file for file in sorted(valid_clip_paths, key=lambda x: int(x.split('/')[-1].split('_')[1]))]
            else:
                return [],[]
        else:
            dense_frame_folder = '/'.join(video_path.split('/')[:-2]) + '/dense_frames/' + video_path.split('/')[-1].split('.')[0] + '/'
            if not os.path.exists(dense_frame_folder) or os.listdir(dense_frame_folder)==[]:
                self.extract_frames(video_path, dense_frame_folder, fps=self.args.clip_fps)
            frame_paths = [dense_frame_folder + file for file in sorted(os.listdir(dense_frame_folder),key = lambda x:float(x.split('/')[-1].split('_')[1].split('.')[0]))]

            clip_frames_li, clip_video_paths = [], []
            pointer = 0
            for clip_id in range(chunk_number):
                cur_frame_li = []
                for frame in frame_paths[pointer:]:
                    cur_seconds = float(frame.split('/')[-1].split('.')[0].split('_')[-1])
                    if cur_seconds>=clip_id*self.args.clip_duration+self.args.clip_duration:
                        break
                    if clip_id*self.args.clip_duration<=cur_seconds<clip_id*self.args.clip_duration+self.args.clip_duration:
                        cur_frame_li.append(frame)
                
                if cur_frame_li==[]:
                    continue 
                pointer += len(cur_frame_li)
                if len(cur_frame_li)<8:
                    cur_frame_li = cur_frame_li + [cur_frame_li[-1]]*(8-len(cur_frame_li))
                step = len(cur_frame_li)//8
                cur_frame_li = cur_frame_li[::step][:8]
                clip_frames_li.append(cur_frame_li)
                start_time = self.args.clip_duration * clip_id
                end_time = start_time + self.args.clip_duration
                output_filename = f'clip_{clip_id}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4'  
                clip_video_paths.append(clip_save_folder+'/'+output_filename) 
            return clip_frames_li, clip_video_paths
        
    
    @ torch.no_grad()
    def calculate_video_clip_embedding(self, video_path, folder_path, total_duration=None, pre_calculate=False):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]

        folder_path = self._clip_embedding_folder()
        os.makedirs(folder_path,exist_ok=True)

        embedding_path = os.path.join(folder_path,video_name+'.pkl')
        clip_path_li = os.path.join(folder_path,video_name+'_clip_paths.pkl')

        if os.path.exists(embedding_path) and os.path.exists(clip_path_li):
            video_paths = pickle.load(open(clip_path_li,'rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))

            if len(video_paths) > total_duration // self.args.clip_duration - 3:
                return video_paths, total_embeddings
            else:
                print(embedding_path,'exist but have not enough valid video number!!')  
        
        if not pre_calculate:
            print('All video clip embedding should be calculated when pre-processing the dataset!')
            return [], []
        
        frames_li, video_path_names  = self.cut_video(video_path, os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0]),total_duration)
        if frames_li != []:
            video_paths = frames_li.copy()
        else:
            video_paths = video_path_names.copy()
        
        
        if len(video_paths) == 0:
            print(f'No valid clips found for {video_path}, skipping...')
            return [], []
        
        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(video_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_video_paths = []
        for i in tqdm(range(len(video_paths)),desc=f'calculating video embedding: {video_name}'):
        # try:
            inputs = {'video': to_device(self.modality_transform['video'](video_paths[i]), self.device)}
            with torch.no_grad():
                embeddings = self.model(inputs)
                valid_video_paths.append(video_path_names[i])
                total_embeddings.append(embeddings['video'])
            # except Exception as e:
            #     print(e)
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_video_paths,open(f'{folder_path}/{video_name}_clip_paths.pkl','wb'))
        return video_paths,total_embeddings




         
    @ torch.no_grad()
    def calculate_clip_embedding_from_frames(self, frame_paths):
        total_embeddings = []  
        inputs = {'video': to_device(self.modality_transform['video'](frame_paths), self.device)}
        with torch.no_grad():
            embeddings = self.model(inputs)
            torch.cuda.empty_cache()
            return embeddings['video']
            



    def extract_frames(self, video_path, output_dir, fps=1):
        os.makedirs(output_dir, exist_ok=True)
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print(f"Failed to open {video_path}")
            return
        
        frame_rate = vid.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print(f"Failed to get FPS for {video_path}")
            return
        
        frame_interval = math.floor(frame_rate / fps)
        frame_idx = 0
        second = 0
        
        with tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), desc=os.path.basename(video_path)) as pbar:
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_filename = os.path.join(output_dir, f"frame_{second}.png")
                    cv2.imwrite(frame_filename, frame)
                    second += 1
                
                frame_idx += 1
                pbar.update(1)
        
        vid.release()

    @ torch.no_grad()
    def calculate_frame_embedding(self, video_path, folder_path, total_duration):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            os.makedirs(f'{args.dataset_folder}/embeddings/frame/{self.args.retriever_type}/',exist_ok=True)
            frame_paths = pickle.load(open(f'{folder_path}/{video_name}_frame_paths.pkl','rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))
            invalid_num=0
            for v in frame_paths:
                if not is_valid_video(v):
                    invalid_num+=1

            if invalid_num<5:
                return frame_paths,total_embeddings
            
        frame_folder = '/'.join(video_path.split('/')[:-2]) + '/dense_frames/' + video_path.split('/')[-1].split('.')[0] + '/'
        if not os.path.exists(frame_folder) or os.listdir(frame_folder)==[]:
            self.extract_frames(video_path, frame_folder, fps=1)
        frame_paths = [frame_folder + file for file in sorted(os.listdir(frame_folder),key = lambda x:float(x.split('/')[-1].split('_')[1].split('.')[0]))]

        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(frame_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_frame_paths = []
        for i in range(len(frame_paths)):
            try:
                inputs = {'image': to_device(self.modality_transform['image'](frame_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_frame_paths.append(frame_paths[i])
                    total_embeddings.append(embeddings['image'])
            except:
                pass
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_frame_paths,open(f'{folder_path}/{video_name}_frame_paths.pkl','wb'))
        return frame_paths,total_embeddings



    @ torch.no_grad()
    def calculate_video_embedding(self, video_path, folder_path):
        video_name = video_path.split('/')[-1].split('.')[0]
        os.makedirs(folder_path,exist_ok=True)
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            try:
                embedding = pickle.load(open(embedding_path,'rb'))
                return embedding
            except:
                pass

        try: 
            inputs = {'video': to_device(self.modality_transform['video'](video_path), self.device)}
            with torch.no_grad():
                embedding = self.model(inputs)
            pickle.dump(embedding,open(f'{folder_path}/{video_name}.pkl','wb'))
            return embedding
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()



    @ torch.no_grad()
    def calculate_text_embedding(self,text,video_path=None,flag_save_embedding=True, modality='video'):
        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                embeddings = pickle.load(open(embedding_path,'rb'))
                return embeddings
            except:
                pass

        language_key = 'language' if modality == getattr(self.model, 'default_text_modality', 'video') else f'language:{modality}'
        inputs = {
            language_key: to_device(
                self.tokenizer(text, max_length=77, padding='max_length',truncation=True, return_tensors='pt'),
                self.device,
            )
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
        if flag_save_embedding:
            pickle.dump(embeddings[language_key],open(embedding_path,'wb'))
        torch.cuda.empty_cache()
        return embeddings[language_key].cpu()


    @ torch.no_grad()
    def calculate_subtitle_embedding(self,video_path,flag_save_embedding=False,merge_sentence=False):
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]
        subtitle_embs = self.calculate_text_embedding(subtitles,video_path,flag_save_embedding=True)
        subtitle_embs = subtitle_embs.cpu()
        return subtitles_with_time,subtitle_embs


    @ torch.no_grad()
    def get_informative_subtitles(self, query, video_path, top_k=1, total_duration=-1, return_embeddings=False,merge_sentence=False,flag_save_embedding=1):
        if not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','.srt')) and not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','_en.json')):
            return ''

        q_emb = self.text_retriever.encode(query, batch_size=12, max_length=256)['dense_vecs']
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]

        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                subtitle_embeddings = pickle.load(open(embedding_path,'rb'))
            except Exception as e:
                subtitle_embeddings = self.text_retriever.encode(subtitles, batch_size=12, max_length=256)['dense_vecs']
                if flag_save_embedding:
                    pickle.dump(subtitle_embeddings,open(embedding_path,'wb'))

        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        return [subtitles_with_time[i] for i in top_k_indices]



    def subtitle2clips(self, subtitle_triple, video_path):
        def is_overlap(begin1, end1, begin2, end2):
            return begin1 <= end2 and begin2 <= end1

        subtitle_begin_time, subtitle_end_time = subtitle_triple[0], subtitle_triple[1]
        ans = []
        for clip in os.listdir(self.clip_save_folder + video_path.split('/')[-1][:-4]):
            clip_begin_time, clip_end_time = self.parse_time(clip.split('.')[0].split('_')[2]),self.parse_time(clip.split('.')[0].split('_')[4])
            if is_overlap(subtitle_begin_time, subtitle_end_time, clip_begin_time, clip_end_time):
                video_clip_path = self.clip_save_folder + video_path.split('/')[-1][:-4] +f'/{clip}'
                ans.append(video_clip_path)
        return ans

    @ torch.no_grad()
    def get_informative_clips_with_video_query(self,query, query_video_path,video_path,top_k=0, total_duration=-1,similarity_threshold=-100,topk_similarity=0, return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        text_emb = self.calculate_text_embedding(query,flag_save_embedding=False, modality='video').cpu()
        text_emb = text_emb / text_emb.norm(p=2, dim=1, keepdim=True)

        inputs = {'video': to_device(self.modality_transform['video'](query_video_path), self.device)}
        with torch.no_grad():
            q_emb = self.model(inputs)['video'].cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        q_emb = q_emb + text_emb

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = self._clip_embedding_folder()
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        similarities = torch.matmul(q_emb, clip_embs.T)

        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10]
        return result



    @ torch.no_grad()
    def get_informative_clips(self,query,video_path,top_k=0,total_duration=-1,similarity_threshold=-100,topk_similarity=0,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        q_emb = self.calculate_text_embedding(query,flag_save_embedding=False, modality='video').cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = self._clip_embedding_folder()
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        similarities = torch.matmul(q_emb, clip_embs.T)

        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10]

        return result

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
        # frame_123.45.png
        num = base.replace("frame_", "").replace(".png", "")
        return float(num)

    @torch.no_grad()
    def get_informative_dense_frames(
        self,
        query: str,
        video_path: str,
        dataset_folder: str,
        top_k: int = 5,
        total_duration: float = None,
        materialize_if_empty: bool = True,
        dense_sample_fps: float = 24.0,
        embed_batch: int = 8,
    ):
        """Text-to-frame retrieval over all dense PNGs (LanguageBind text vs image embeddings)."""
        torch.cuda.empty_cache()
        if not query or not str(query).strip():
            return []

        frame_paths, dense_dir = self._list_dense_frame_paths(dataset_folder, video_path)
        if not frame_paths and materialize_if_empty and total_duration is not None and total_duration > 0:
            try:
                from video_utils import timestamp_to_clip_path

                timestamp_to_clip_path(
                    dataset_folder,
                    0.0,
                    float(total_duration),
                    video_path,
                    fps=float(dense_sample_fps),
                )
            except Exception as e:
                print(f"  dense frame materialize skipped: {e}")
            frame_paths, dense_dir = self._list_dense_frame_paths(dataset_folder, video_path)

        if not frame_paths:
            return []

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_path = os.path.join(
            dataset_folder, "embeddings", "dense_frame", self.args.retriever_type
        )
        os.makedirs(folder_path, exist_ok=True)
        emb_path = os.path.join(folder_path, f"{video_name}.pkl")
        paths_path = os.path.join(folder_path, f"{video_name}_frame_paths.pkl")

        q_emb = self.calculate_text_embedding(query, flag_save_embedding=False, modality='image').cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        cache_key = os.path.abspath(video_path)
        frame_embs = None
        if cache_key in self.dense_frame_embs_cache:
            cp, fe = self.dense_frame_embs_cache[cache_key]
            if cp == frame_paths:
                frame_embs = fe
            else:
                del self.dense_frame_embs_cache[cache_key]

        if frame_embs is None and os.path.exists(emb_path) and os.path.exists(paths_path):
            try:
                cp = pickle.load(open(paths_path, "rb"))
                frame_embs = pickle.load(open(emb_path, "rb"))
                if isinstance(frame_embs, dict):
                    frame_embs = frame_embs.get("image", frame_embs)
                if cp == frame_paths:
                    self.dense_frame_embs_cache[cache_key] = (frame_paths, frame_embs)
                else:
                    frame_embs = None
            except Exception:
                frame_embs = None

        if frame_embs is None:
            total_embeddings = []
            valid_paths = []
            batch_size = max(1, int(embed_batch))
            for i in tqdm(
                range(0, len(frame_paths), batch_size),
                desc=f"dense_frame_emb {video_name}",
            ):
                batch = frame_paths[i : i + batch_size]
                try:
                    inputs = {
                        "image": to_device(
                            self.modality_transform["image"](batch), self.device
                        )
                    }
                    with torch.no_grad():
                        emb = self.model(inputs)["image"].cpu()
                    total_embeddings.append(emb)
                    valid_paths.extend(batch)
                except Exception:
                    for p in batch:
                        try:
                            inputs = {
                                "image": to_device(
                                    self.modality_transform["image"](p), self.device
                                )
                            }
                            with torch.no_grad():
                                emb = self.model(inputs)["image"].cpu()
                            total_embeddings.append(emb)
                            valid_paths.append(p)
                        except Exception as e2:
                            print(f"  skip frame emb {p}: {e2}")
                    torch.cuda.empty_cache()

            if not total_embeddings:
                return []
            frame_embs = torch.cat(total_embeddings, dim=0)
            if len(valid_paths) != len(frame_paths):
                frame_paths = valid_paths
            os.makedirs(folder_path, exist_ok=True)
            pickle.dump(frame_paths, open(paths_path, "wb"))
            pickle.dump(frame_embs, open(emb_path, "wb"))
            self.dense_frame_embs_cache[cache_key] = (frame_paths, frame_embs)

        frame_embs = frame_embs.cpu()
        frame_embs = frame_embs / frame_embs.norm(p=2, dim=1, keepdim=True)
        similarities = torch.matmul(q_emb, frame_embs.T)[0]
        k = min(top_k, similarities.shape[0])
        top_k_indices = similarities.argsort(descending=True)[:k].tolist()

        result = []
        for i in top_k_indices:
            result.append((frame_paths[i], similarities[i].item()))
        torch.cuda.empty_cache()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlvu')
    parser.add_argument('--dataset_mode', type=str, default='')
    parser.add_argument('--dataset_folder', type=str, default='')
    parser.add_argument('--read_path', type=str)
    parser.add_argument('--clip_duration', type=int, default=4)
    parser.add_argument('--topk_per_query', type=int, default=1)
    parser.add_argument('--retriever_type', type=str, default='large')
    parser.add_argument('--thread_idx', type=int, default=0)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--begin_sample_number', type=int, default=0)
    parser.add_argument('--end_sample_number', type=int, default=10000000000)
    parser.add_argument('--random_shuffle', action='store_true')
    parser.add_argument('--overwrite_output', type=int, default=0)
    parser.add_argument('--use_subtitle', type=int, default=0)
    parser.add_argument('--use_vllm', type=int, default=1)
    parser.add_argument('--clip_fps', type=float, default=2)
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--max_workers', type=int, default=1) 
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--mix_clip_duration', type=str, default='10')

    args = parser.parse_args()

    a = ['./benchmark/NIAH-Video/dense_frames/video_haystack/frame_2091.52.png','./benchmark/NIAH-Video/dense_frames/video_haystack/frame_2160.68.png']
    a = a+a+a+a
    
    retriever =  Retrieval_Manager(args, clip_save_folder='./benchmark/NIAH-Video/clips/')
    
    retriever.load_model_to_gpu(0)
    embedding = retriever.calculate_clip_embedding_from_frames(a)
    
    
