import json
import re

from video_utils import extract_subtitles, robust_eval, timestamp_to_clip_path


class RefinerUtilsMixin:
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
