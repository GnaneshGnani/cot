import json
import re

from video_utils import extract_subtitles, robust_eval, timestamp_to_clip_path


def openai_chat_completion_limit_kwargs(model_name: str, limit: int) -> dict:
    """Kwargs for OpenAI ``chat.completions.create`` output length.

    Models such as **gpt-5** require ``max_completion_tokens``; older models use ``max_tokens``.
    """
    mn = (model_name or "").strip().lower()
    if mn.startswith("gpt-5") or mn.startswith(("o1", "o2", "o3", "o4")):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def openai_chat_temperature_kwargs(model_name: str, temperature: float) -> dict:
    """Kwargs for ``temperature``. gpt-5 and some reasoning models only allow the API default (omit param)."""
    mn = (model_name or "").strip().lower()
    if mn.startswith("gpt-5") or mn.startswith(("o1", "o2", "o3", "o4")):
        return {}
    return {"temperature": temperature}


class RefinerUtilsMixin:
    def _safe_float(self, value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # Matches all planner-generated reference variants:
    #   <STEP_1:frames[0].frame_path>  (canonical)
    #   <STEP1:frames[0].frame_path>   (no underscore)
    #   <STEP_1.frame_path>            (dot instead of colon, abbreviated path)
    #   <STEP1.frame_path>             (no underscore, dot separator)
    _STEP_REF_RE = re.compile(r"<STEP_?(\d+)[:\.]([^>]+)>")

    def _traverse_json_path(self, obj, path: str):
        """Traverse dot-and-bracket notation on a parsed JSON object.

        E.g. 'frames[0].frame_path' on {"frames": [{"frame_path": "..."}]}
        returns the string value.
        """
        for token in re.split(r"\.", path):
            if obj is None:
                return None
            m = re.match(r"^(\w+)\[(\d+)\]$", token)
            if m:
                obj = obj.get(m.group(1)) if isinstance(obj, dict) else None
                if isinstance(obj, list):
                    idx = int(m.group(2))
                    obj = obj[idx] if idx < len(obj) else None
                else:
                    obj = None
            else:
                obj = obj.get(token) if isinstance(obj, dict) else None
        return obj

    def _resolve_json_ref(self, obj, path: str):
        """Traverse path on obj with a fallback for abbreviated paths.

        If the path is a simple field name (no dots or brackets) and is not found
        at the root, searches the first element of each top-level list. This handles
        planner shorthands like `frame_path` that should be `frames[0].frame_path`.
        """
        if obj is None:
            return None
        result = self._traverse_json_path(obj, path)
        if result is not None:
            return result
        if "." not in path and "[" not in path and isinstance(obj, dict):
            for val in obj.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    result = self._traverse_json_path(val[0], path)
                    if result is not None:
                        return result
        return None

    def _resolve_step_refs(self, value, step_results: dict):
        """Recursively substitute <STEP_N:json.path> templates with their resolved values.

        step_results maps step number (int) to the parsed JSON output dict of that step.
        If the entire string is a template, the resolved native value (any type) is returned.
        Embedded templates inside a larger string are replaced with their str() representation.
        Unresolvable references are left unchanged.
        """
        if isinstance(value, str):
            full = self._STEP_REF_RE.fullmatch(value.strip())
            if full:
                resolved = self._resolve_json_ref(
                    step_results.get(int(full.group(1))), full.group(2)
                )
                return resolved if resolved is not None else value

            def _sub(m):
                resolved = self._resolve_json_ref(
                    step_results.get(int(m.group(1))), m.group(2)
                )
                return str(resolved) if resolved is not None else m.group(0)

            return self._STEP_REF_RE.sub(_sub, value)
        if isinstance(value, dict):
            return {k: self._resolve_step_refs(v, step_results) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_step_refs(item, step_results) for item in value]
        return value

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
                candidates.append(text[start : end + 1])

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

    def _refiner_dense_frame_fps(self) -> float:
        """Sampling rate for on-disk dense frames (seconds between samples ≈ 1/fps)."""
        fps = float(getattr(self, "dense_frame_fps", 24.0))
        return max(0.1, min(fps, 120.0))

    def _get_frames_for_range(self, start_time=None, end_time=None, fps=None):
        if fps is None:
            fps = self._refiner_dense_frame_fps()
        start, end = self._get_time_range(start_time, end_time)
        frame_paths, timestamps = timestamp_to_clip_path(
            self.dataset_folder, start, end, self.video_path, fps=fps
        )
        return frame_paths, timestamps, start, end

    def _get_frame_at_timestamp(self, timestamp: float):
        timestamp = self._safe_float(timestamp, 0.0)
        frame_paths, timestamps, _, _ = self._get_frames_for_range(timestamp, timestamp, fps=None)
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

    def _extract_final_answer(self, text: str) -> str:
        try:
            answer_content = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)[-1].strip()
            answer_content = re.sub(r"\s+", " ", answer_content)
            return answer_content if answer_content else "-"
        except Exception:
            return "-"

    def _normalize_answer(self, answer: str) -> str:
        return re.sub(r"\s+", " ", str(answer)).strip().lower()

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
