import json
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from video_utils import extract_subtitles, robust_eval, timestamp_to_clip_path


_UNRESOLVED_STEP_REF_RE = re.compile(r"<STEP_?\d+[:\.][^>]+>")
_UNQUOTED_PLACEHOLDER_RE = re.compile(r'(?<!["\'])<[^<>\n]+>(?!["\'])')


def _coerce_float_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        parsed = robust_eval(value)
        value = parsed if parsed != value else [value]
    elif isinstance(value, (int, float)):
        value = [value]
    elif not isinstance(value, list):
        value = [value]
    return [float(item) for item in value]


class FrameBundleItemModel(BaseModel):
    frame_path: str
    timestamp: float


class TemporalGrounderArgsModel(BaseModel):
    query: str
    video_path: Optional[str] = None

    @validator("query")
    def _validate_query(cls, value):
        value = str(value or "").strip()
        if not value:
            raise ValueError("query must be non-empty")
        return value


class FrameRetrieverArgsModel(BaseModel):
    video_path: Optional[str] = None
    query: Optional[str] = None
    timestamps: Optional[List[float]] = None
    num_frames: int = 5

    @validator("timestamps", pre=True)
    def _normalize_timestamps(cls, value):
        return _coerce_float_list(value)

    @validator("num_frames")
    def _validate_num_frames(cls, value):
        if int(value) < 1:
            raise ValueError("num_frames must be at least 1")
        return int(value)

    @root_validator(skip_on_failure=True)
    def _validate_source(cls, values):
        query = str(values.get("query") or "").strip()
        timestamps = values.get("timestamps") or []
        if not query and not timestamps:
            raise ValueError("frame_retriever requires either query or timestamps")
        values["query"] = query or None
        return values


class ChartAnalyzerArgsModel(BaseModel):
    frame_path: Optional[Union[str, List[str], List[FrameBundleItemModel]]] = None
    timestamp: Optional[Union[float, List[float]]] = None
    query: Optional[str] = None

    @validator("timestamp", pre=True)
    def _normalize_timestamp(cls, value):
        if value in (None, [], ""):
            return None
        if isinstance(value, dict):
            value = value.get("timestamp")
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            value = [item.get("timestamp") for item in value]
        if isinstance(value, list):
            return [float(item) for item in value]
        return float(value)

    @validator("query")
    def _normalize_query(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class VLMFrameBundleModel(BaseModel):
    frame_paths: List[str] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)

    @validator("frame_paths")
    def _validate_frame_paths(cls, value):
        if not value:
            raise ValueError("frame_paths must not be empty")
        cleaned = []
        for item in value:
            path = str(item or "").strip()
            if not path:
                raise ValueError("frame_paths must contain non-empty strings")
            cleaned.append(path)
        return cleaned

    @validator("timestamps", pre=True)
    def _normalize_vlm_timestamps(cls, value):
        coerced = _coerce_float_list(value)
        return coerced or []

    @root_validator(skip_on_failure=True)
    def _validate_bundle(cls, values):
        frame_paths = values.get("frame_paths") or []
        timestamps = values.get("timestamps") or []
        if len(frame_paths) != len(timestamps):
            raise ValueError("frame_paths and timestamps must have the same length")
        if len(frame_paths) > 1:
            if any(curr <= prev for prev, curr in zip(timestamps, timestamps[1:])):
                raise ValueError("multi-frame VLM inputs require strictly increasing timestamps")
            if len({round(ts, 6) for ts in timestamps}) < 2:
                raise ValueError("multi-frame VLM inputs require distinct timestamps")
        return values


class PlannerToolCallModel(BaseModel):
    step: int
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""
    depends_on: List[int] = Field(default_factory=list)

    @validator("step", pre=True)
    def _normalize_step(cls, value):
        return int(value)

    @validator("tool", pre=True)
    def _normalize_tool(cls, value):
        value = str(value or "").strip()
        if not value:
            raise ValueError("tool must be non-empty")
        return value

    @validator("arguments", pre=True)
    def _normalize_arguments(cls, value):
        return value if isinstance(value, dict) else {}

    @validator("purpose", pre=True)
    def _normalize_purpose(cls, value):
        return str(value or "").strip()

    @validator("depends_on", pre=True)
    def _normalize_depends_on(cls, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            value = [value]
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out


class PlannerOutputModel(BaseModel):
    strategy: str = ""
    tool_calls: List[PlannerToolCallModel] = Field(default_factory=list)
    refinement_instructions: str = ""

    @validator("strategy", "refinement_instructions", pre=True)
    def _normalize_text(cls, value):
        return str(value or "").strip()


class VerifierTraceQualityScoresModel(BaseModel):
    perceptual_correctness: float = 0.0
    temporal_accuracy: float = 0.0
    logical_coherence: float = 0.0
    completeness: float = 0.0


class VerifierErrorCategoryModel(BaseModel):
    type: str = ""
    step_index: Optional[int] = None
    description: str = ""
    severity: str = "LOW"
    suggested_tools: List[str] = Field(default_factory=list)
    evidence: Any = None

    @validator("step_index", pre=True)
    def _normalize_step_index(cls, value):
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @validator("type", "description", "severity", pre=True)
    def _normalize_string_fields(cls, value):
        return str(value or "").strip()

    @validator("suggested_tools", pre=True)
    def _normalize_suggested_tools(cls, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]


class VerifierOutputModel(BaseModel):
    verdict: str = "FAIL"
    answer_correct: bool = False
    trace_quality_scores: VerifierTraceQualityScoresModel = Field(
        default_factory=VerifierTraceQualityScoresModel
    )
    error_categories: List[VerifierErrorCategoryModel] = Field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""

    @validator("verdict", pre=True)
    def _normalize_verdict(cls, value):
        value = str(value or "FAIL").strip().upper()
        return value if value in {"PASS", "FAIL"} else "FAIL"

    @validator("summary", pre=True)
    def _normalize_summary(cls, value):
        return str(value or "").strip()


class RefinerChangeModel(BaseModel):
    operation: str = ""
    step_index: Optional[int] = None
    original: str = ""
    replacement: str = ""
    evidence_source: str = ""

    @validator("step_index", pre=True)
    def _normalize_change_step_index(cls, value):
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @validator("operation", "original", "replacement", "evidence_source", pre=True)
    def _normalize_change_fields(cls, value):
        return str(value or "").strip()


class RefinerOutputModel(BaseModel):
    refined_trace: Optional[Union[str, List[str]]] = None
    refined_answer: Optional[str] = None
    answer_changed: bool = False
    changes_made: List[RefinerChangeModel] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)

    @validator("refined_trace", pre=True)
    def _normalize_refined_trace(cls, value):
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        return str(value)

    @validator("refined_answer", pre=True)
    def _normalize_refined_answer(cls, value):
        if value is None:
            return None
        return str(value).strip() or None

    @validator("unresolved_issues", pre=True)
    def _normalize_unresolved_issues(cls, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]


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

    def _ensure_no_unresolved_step_refs(self, value, path: str = "arguments"):
        if isinstance(value, str):
            if _UNRESOLVED_STEP_REF_RE.search(value):
                raise ValueError(f"{path} contains unresolved step reference: {value}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._ensure_no_unresolved_step_refs(item, f"{path}.{key}")
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                self._ensure_no_unresolved_step_refs(item, f"{path}[{idx}]")

    def _validate_tool_arguments(self, tool_name: str, arguments: dict) -> dict:
        if not isinstance(arguments, dict):
            raise ValueError(f"{tool_name} arguments must be a JSON object")

        self._ensure_no_unresolved_step_refs(arguments)

        model_map = {
            "temporal_grounder": TemporalGrounderArgsModel,
            "frame_retriever": FrameRetrieverArgsModel,
            "chart_analyzer": ChartAnalyzerArgsModel,
        }
        model_cls = model_map.get(tool_name)
        if model_cls is None:
            return arguments

        validated = model_cls.parse_obj(arguments)
        return validated.dict(exclude_none=True)

    def _tool_validation_error_result(self, tool_name: str, error: Exception) -> dict:
        return {
            "ok": False,
            "error": f"{tool_name} argument validation failed: {error}",
        }

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
        m = re.match(r"^\[(\d+)\](?:\.(.+))?$", path)
        if m:
            idx = m.group(1)
            rest = m.group(2)
            path = f"{idx}.{rest}" if rest else idx
        result = self._traverse_json_path(obj, path)
        if result is not None:
            return result
        m = re.match(r"^(\d+)\.(.+)$", path)
        if m and isinstance(obj, dict):
            idx = int(m.group(1))
            rest = m.group(2)
            for val in obj.values():
                if isinstance(val, list) and idx < len(val):
                    result = self._traverse_json_path(val[idx], rest)
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
        return self._extract_json_payload_with_schema(text)

    def _repair_json_candidate(self, candidate: str) -> str:
        if not isinstance(candidate, str) or "<" not in candidate:
            return candidate
        return _UNQUOTED_PLACEHOLDER_RE.sub(lambda m: json.dumps(m.group(0)), candidate)

    def _repair_truncated_json_candidate(self, candidate: str) -> str:
        if not isinstance(candidate, str):
            return candidate

        text = candidate.strip()
        if not text:
            return text

        starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
        if starts:
            text = text[min(starts) :]

        out = []
        stack = []
        in_string = False
        escape = False

        for ch in text:
            out.append(ch)
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()

        repaired = "".join(out).rstrip()
        if in_string:
            trailing_backslashes = 0
            for ch in reversed(repaired):
                if ch == "\\":
                    trailing_backslashes += 1
                else:
                    break
            if trailing_backslashes % 2 == 1:
                repaired += "\\"
            repaired += '"'

        repaired = repaired.rstrip()
        if repaired.endswith(":"):
            repaired += " null"
        while repaired.endswith(","):
            repaired = repaired[:-1].rstrip()

        repaired += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        return repaired

    def _validate_json_payload(self, payload, model_cls=None):
        if model_cls is None:
            return payload
        try:
            validated = model_cls.parse_obj(payload)
        except ValidationError:
            return None
        return validated.dict(exclude_none=False)

    def _extract_json_payload_with_schema(self, text, model_cls=None, repair_placeholders: bool = False):
        if isinstance(text, (dict, list)):
            return self._validate_json_payload(text, model_cls)
        if not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        candidates.append(text)
        starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
        if starts:
            candidates.append(text[min(starts) :])

        for left, right in (("{", "}"), ("[", "]")):
            start = text.find(left)
            end = text.rfind(right)
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start : end + 1])

        for candidate in candidates:
            variants = [candidate]
            truncated = self._repair_truncated_json_candidate(candidate)
            if truncated != candidate:
                variants.append(truncated)
            if repair_placeholders:
                repaired_variants = []
                for variant in list(variants):
                    repaired = self._repair_json_candidate(variant)
                    if repaired != variant:
                        repaired_variants.append(repaired)
                variants.extend(repaired_variants)
            deduped_variants = []
            seen = set()
            for variant in variants:
                key = str(variant)
                if key in seen:
                    continue
                seen.add(key)
                deduped_variants.append(variant)
            for variant in deduped_variants:
                variant = variant.strip()
                if not variant:
                    continue
                try:
                    parsed = json.loads(variant)
                except Exception:
                    parsed = robust_eval(variant)
                if isinstance(parsed, (dict, list)):
                    validated = self._validate_json_payload(parsed, model_cls)
                    if validated is not None:
                        return validated
        return None

    def _extract_planner_payload(self, text):
        return self._extract_json_payload_with_schema(
            text,
            model_cls=PlannerOutputModel,
            repair_placeholders=True,
        )

    def _extract_verifier_payload(self, text):
        return self._extract_json_payload_with_schema(
            text,
            model_cls=VerifierOutputModel,
            repair_placeholders=False,
        )

    def _extract_refiner_payload(self, text):
        return self._extract_json_payload_with_schema(
            text,
            model_cls=RefinerOutputModel,
            repair_placeholders=False,
        )

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

    def _run_vlm_json(self, prompt: str, frame_paths: list, timestamps: list, default_result, force_local: bool = False):
        if not frame_paths:
            return default_result

        try:
            validated = VLMFrameBundleModel.parse_obj(
                {"frame_paths": frame_paths, "timestamps": timestamps}
            )
            frame_paths = list(validated.frame_paths)
            timestamps = list(validated.timestamps)
        except ValidationError as e:
            if isinstance(default_result, dict):
                result = dict(default_result)
                result["raw_output"] = f"Invalid VLM frame bundle: {e}"
                return result
            return default_result

        output_text = self._batch_video2text([(prompt, frame_paths, timestamps)], force_local=force_local)[0]
        parsed = self._extract_json_payload(output_text)
        if parsed is not None:
            return parsed

        if isinstance(default_result, dict):
            result = dict(default_result)
            result["raw_output"] = output_text
            return result
        return default_result

    def _get_asr_result_from_subtitles(self, start_time=None, end_time=None):
        subtitle_error = None
        try:
            subtitle_segments = extract_subtitles(self.video_path)
        except Exception as ex:
            subtitle_error = f"{type(ex).__name__}: {ex}"
            subtitle_segments = []

        requested_range = None
        if start_time is None and end_time is None:
            filtered = subtitle_segments
        else:
            start, end = self._get_time_range(start_time, end_time)
            requested_range = {"start": float(start), "end": float(end)}
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
            "subtitle_source_available": bool(subtitle_segments),
            "requested_range": requested_range,
            "subtitle_error": subtitle_error,
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
