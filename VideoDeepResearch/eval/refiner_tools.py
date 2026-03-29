"""
Refinement tool handlers (temporal grounder, OCR, ASR, etc.) as a mixin on VideoQADemo.
"""
import json
import os
import subprocess
import tempfile

from PIL import Image

from video_utils import robust_eval
from refine_prompt import (
    action_recognizer_prompt,
    counter_prompt,
    dense_captioner_prompt,
    ocr_prompt,
    spatial_grunder_prompt,
    video_qa_reanswerer_prompt,
)

# Lazy singletons for optional heavy backends
_WHISPERX_MODEL = None
_WHISPERX_ALIGN = None
_WHISPERX_META = None
_PADDLE_OCR = None
_CLAP_MODULE = None


class RefinerToolsMixin:
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
        for call in ordered_calls:
            tool_name = call.get("tool", "")
            arguments = call.get("arguments", {})
            output = self._execute_refine_tool_call(
                tool_name, arguments if isinstance(arguments, dict) else {}
            )
            execution_results.append(
                {
                    "step": call.get("step"),
                    "tool": tool_name,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                    "purpose": call.get("purpose", ""),
                    "depends_on": call.get("depends_on", []),
                    "output": output.strip(),
                }
            )
        return execution_results

    def _get_asr_whisperx(self, start_time=None, end_time=None):
        global _WHISPERX_MODEL, _WHISPERX_ALIGN, _WHISPERX_META
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_name = os.getenv("WHISPERX_MODEL", "large-v3")

        import whisperx

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
        topk = int(os.getenv("TOPK", "5"))

        for arguments in calls:
            query = str(arguments.get("query", "")).strip()
            segments = []
            if query:
                try:
                    clip_results = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, top_k=topk, total_duration=self.duration
                    )
                    for clip_path, score in clip_results:
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
                    clip_results = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, top_k=num_frames, total_duration=self.duration
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    clip_results = []

                for clip_path, score in clip_results[:num_frames]:
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
        import numpy as np
        import torch
        import soundfile as sf

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

        try:
            from laion_clap import CLAP_Module
        except ImportError:
            return {
                "query": query,
                "events": [],
                "audio_summary": "laion_clap not installed",
                "backend": "none",
            }

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
        from paddleocr import PaddleOCR

        if _PADDLE_OCR is None:
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
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
        import pytesseract

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

            if not frame_path:
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
            if not frame_path:
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
                arguments.get("start_time"), arguments.get("end_time"), fps=2.0
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
                video_qa_reanswerer_prompt.strip()
                + f"\n\nQuestion: {question}\nReturn JSON only.\n"
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
