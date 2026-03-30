import base64
import io
import json
import os
import subprocess
import sys
import tempfile
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


class RefinerToolsMixin:
    def _chart_torch_device(self):
        raw = str(getattr(self, "chart_device", "cuda:1")).strip()
        if raw.isdigit():
            return torch.device(f"cuda:{int(raw)}")
        return torch.device(raw)

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

        print("\n" + "=" * 70)
        print("ordered_calls: ", ordered_calls)
        print("=" * 70 + "\n")

        execution_results = []
        step_results: dict = {}  # step_num (int) → parsed JSON output for dep resolution
        ibase = getattr(self, "_refinement_debug_iter_dir", None)

        for call in ordered_calls:
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            args_dict = arguments if isinstance(arguments, dict) else {}

            # Resolve any <STEPN:json.path> references from prior step outputs
            args_dict = self._resolve_step_refs(args_dict, step_results)

            tool_out_dir = None
            if ibase:
                step = int(call.get("step") or 0)
                slug = refiner_debug.sanitize_path_component(str(tool_name))
                tbase = Path(ibase) / f"tool_{step:02d}_{slug}"
                tool_out_dir = refiner_debug.ensure_outputs_dir(tbase)
                refiner_debug.write_json(tool_out_dir, "arguments.json", args_dict)
                self._refinement_debug_vlm_outputs_dir = tool_out_dir
                self._refinement_debug_vlm_input_basename = "model_input.json"

            try:
                output = self._execute_refine_tool_call(tool_name, args_dict)
            finally:
                self._refinement_debug_vlm_outputs_dir = None
                self._refinement_debug_vlm_input_basename = None

            if tool_out_dir:
                refiner_debug.write_text(tool_out_dir, "output.txt", (output or "").strip())

            # Store parsed output so later steps can reference it via <STEPN:...>
            step_num = int(call.get("step") or 0)
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
                    "depends_on": call.get("depends_on", []),
                    "output": (output or "").strip(),
                }
            )
        return execution_results

    def _get_asr_whisperx(self, start_time=None, end_time=None):
        global _WHISPERX_MODEL, _WHISPERX_ALIGN, _WHISPERX_META
        if whisperx is None or torch is None:
            raise ImportError("whisperx or torch not installed")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_name = os.getenv("WHISPERX_MODEL", "large-v3")

        if _WHISPERX_MODEL is None:
            _WHISPERX_MODEL = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(self.video_path)
        result = _WHISPERX_MODEL.transcribe(audio, batch_size=int(os.getenv("WHISPERX_BATCH", "8")))
        lang = result.get("language") or "en"
        segs = []
        try:
            if _WHISPERX_ALIGN is None or _WHISPERX_META is None:
                model_a, meta = whisperx.load_align_model(language_code=lang, device=device)
                _WHISPERX_ALIGN, _WHISPERX_META = model_a, meta
            aligned = whisperx.align(
                result["segments"],
                _WHISPERX_ALIGN,
                _WHISPERX_META,
                audio,
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
        start, end = self._get_time_range(start_time, end_time)
        filtered = [s for s in segs if float(s.get("end", 0)) >= start and float(s.get("start", 0)) <= end]
        segments = []
        words_flat = []
        for s in filtered:
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
            if os.getenv("REFINER_DISABLE_WHISPERX", "").strip() != "1":
                try:
                    result = self._get_asr_whisperx(arguments.get("start_time"), arguments.get("end_time"))
                except Exception as e:
                    print(f"  WhisperX ASR fallback: {e}")
            if result is None:
                result = self._get_asr_result_from_subtitles(
                    arguments.get("start_time"), arguments.get("end_time")
                )
                result["asr_backend"] = result.get("asr_backend", "subtitles")
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
            frame_path = arguments.get("frame_path")
            frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)

            if not frame_path or not os.path.exists(frame_path):
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            source = frame_path if frame_path else f"{self.video_path}@{frame_ts}"
            result = {"source": source, "detections": [], "full_text": "", "ocr_backend": "none"}

            if frame_path and os.path.exists(frame_path):
                detections = []
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
                        + "\n\nExtract all visible text from this frame. Return JSON only.\n"
                    )
                    result = self._run_vlm_json(
                        prompt,
                        [frame_path],
                        [float(frame_ts)],
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
            if not frame_path or not os.path.exists(frame_path):
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            default_result = {
                "query": query,
                "detections": [],
                "spatial_description": "",
            }
            prompt = (
                spatial_grunder_prompt.strip()
                + f"\n\nQuery: {query}\nReturn JSON only matching the OUTPUT FORMAT above.\n"
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
            frame_path = arguments.get("frame_path")
            frame_ts = self._safe_float(arguments.get("timestamp"), 0.0)
            query = str(arguments.get("query", "") or "").strip()

            if not frame_path or not os.path.exists(frame_path):
                frame_path, frame_ts = self._get_frame_at_timestamp(frame_ts)

            default_result = {
                "chart_type": "unknown",
                "title": "",
                "axes": {},
                "series": [],
                "key_observations": [],
                "relationships": [],
                "query_response": None,
            }

            if not frame_path or not os.path.exists(frame_path):
                default_result["query_response"] = "chart_analyzer unavailable or frame not found"
                results.append(self._format_refine_tool_result("chart_analyzer", arguments, default_result))
                continue

            try:
                prompt_text = chart_analyzer_prompt.strip()
                if query:
                    prompt_text += f"\n\nQuery: {query}\nReturn JSON only matching the OUTPUT FORMAT above.\n"
                else:
                    prompt_text += "\n\nReturn JSON only matching the OUTPUT FORMAT above.\n"

                mode = getattr(self, "chart_mode", "api")
                raw_output = None
                parsed = None

                if mode == "api":
                    raw_output = self._call_chart_vision_api(prompt_text, frame_path)
                    parsed = self._extract_json_payload(raw_output)
                elif mode == "vlm":
                    merged = self._run_vlm_json(
                        prompt_text, [frame_path], [float(frame_ts)], default_result
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
