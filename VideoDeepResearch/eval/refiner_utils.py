import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch

from pydantic import BaseModel, Field, ValidationError, parse_obj_as, root_validator, validator

from video_utils import extract_subtitles, robust_eval, timestamp_to_clip_path


_UNRESOLVED_STEP_REF_RE = re.compile(r"<STEP_?\d+[:\.][^>]+>")
_UNQUOTED_PLACEHOLDER_RE = re.compile(r'(?<!["\'])<[^<>\n]+>(?!["\'])')
_VIDEO_PATH_PLACEHOLDERS = {
    "VIDEO",
    "<VIDEO>",
    "VIDEO_PATH",
    "<VIDEO_PATH>",
    "$VIDEO",
    "${VIDEO}",
    "$VIDEO_PATH",
    "${VIDEO_PATH}",
}


def _coerce_float_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        parsed = robust_eval(value)
        value = parsed if parsed != value else [value]
    elif isinstance(value, (int, float)):
        value = [value]
    elif isinstance(value, tuple):
        value = list(value)
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
    time_range: Optional[List[float]] = None
    num_frames: int = 5

    @validator("timestamps", pre=True)
    def _normalize_timestamps(cls, value):
        return _coerce_float_list(value)

    @validator("time_range", pre=True)
    def _normalize_time_range(cls, value):
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
        time_range = values.get("time_range") or []
        if not query and not timestamps:
            raise ValueError("frame_retriever requires either query or timestamps")
        if time_range:
            if len(time_range) != 2:
                raise ValueError("time_range must contain exactly two timestamps")
            start, end = float(time_range[0]), float(time_range[1])
            if end <= start:
                raise ValueError("time_range end must be greater than start")
            values["time_range"] = [start, end]
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


class MathSolverArgsModel(BaseModel):
    question: Optional[str] = None
    evidence: Union[str, List[str]]
    answer_choices: Optional[List[str]] = None
    frame_path: Optional[Union[str, List[str], List[FrameBundleItemModel]]] = None
    timestamp: Optional[Union[float, List[float]]] = None

    @root_validator(pre=True)
    def _alias_evidence(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            if "evidence" not in values and "facts" in values:
                values["evidence"] = values.get("facts")
            if "frame_path" not in values:
                if "frames" in values:
                    values["frame_path"] = values.get("frames")
                elif "frame_paths" in values:
                    values["frame_path"] = values.get("frame_paths")
            if "timestamp" not in values and "timestamps" in values:
                values["timestamp"] = values.get("timestamps")
        return values

    @validator("question", pre=True)
    def _normalize_question(cls, value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @validator("evidence", pre=True)
    def _normalize_evidence(cls, value):
        if value in (None, ""):
            raise ValueError("evidence must be non-empty")
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if not cleaned:
                raise ValueError("evidence must contain at least one non-empty item")
            return cleaned
        value = str(value).strip()
        if not value:
            raise ValueError("evidence must be non-empty")
        return value

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

    @validator("answer_choices", pre=True)
    def _normalize_answer_choices(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, str):
            parsed = robust_eval(value)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [value]
        elif not isinstance(value, list):
            value = [value]
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None


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


class TraceGeneratorOutputModel(BaseModel):
    type: str = ""
    tool: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""
    trace_steps: List[str] = Field(default_factory=list)
    answer: str = ""

    @validator("type", "tool", "purpose", "answer", pre=True)
    def _normalize_generator_string_fields(cls, value):
        return str(value or "").strip()

    @validator("arguments", pre=True)
    def _normalize_generator_arguments(cls, value):
        return value if isinstance(value, dict) else {}

    @validator("trace_steps", pre=True)
    def _normalize_generator_trace_steps(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    @root_validator(skip_on_failure=True)
    def _validate_generator_payload(cls, values):
        output_type = str(values.get("type") or "").strip().lower()
        tool = str(values.get("tool") or "").strip()
        trace_steps = list(values.get("trace_steps") or [])

        if not output_type:
            if trace_steps:
                output_type = "trace"
            elif tool:
                output_type = "tool_call"

        if output_type == "trace":
            if not trace_steps:
                raise ValueError("trace output requires non-empty trace_steps")
        elif output_type == "tool_call":
            if not tool:
                raise ValueError("tool_call output requires non-empty tool")
        else:
            raise ValueError("type must be 'trace' or 'tool_call'")

        values["type"] = output_type
        return values


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
    _UNRESOLVED_TRACE_PATTERNS = [
        re.compile(r"\bambiguous\b", flags=re.IGNORECASE),
        re.compile(r"\bunclear\b", flags=re.IGNORECASE),
        re.compile(r"\bunsupported\b", flags=re.IGNORECASE),
        re.compile(r"\bunresolved\b", flags=re.IGNORECASE),
        re.compile(r"\binsufficient evidence\b", flags=re.IGNORECASE),
        re.compile(r"\bnot enough evidence\b", flags=re.IGNORECASE),
        re.compile(r"\bcannot determine\b", flags=re.IGNORECASE),
        re.compile(r"\bcan't determine\b", flags=re.IGNORECASE),
        re.compile(r"\bunable to determine\b", flags=re.IGNORECASE),
        re.compile(r"\bcannot be determined\b", flags=re.IGNORECASE),
        re.compile(r"\bcannot be verified\b", flags=re.IGNORECASE),
        re.compile(r"\bnot visible\b", flags=re.IGNORECASE),
        re.compile(r"\bnot grounded\b", flags=re.IGNORECASE),
        re.compile(r"\bnot localized\b", flags=re.IGNORECASE),
    ]

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

    def _resolve_step_refs(
        self,
        value,
        step_results: dict,
        fallback_step_results: Optional[Dict[int, Any]] = None,
        blocked_steps: Optional[Set[int]] = None,
    ):
        """Recursively substitute <STEP_N:json.path> templates with their resolved values.

        step_results maps step number (int) to the parsed JSON output dict of that step.
        If the entire string is a template, the resolved native value (any type) is returned.
        Embedded templates inside a larger string are replaced with their str() representation.
        Unresolvable references are left unchanged.
        """
        fallback_step_results = fallback_step_results or {}
        blocked_steps = blocked_steps or set()

        def _lookup(step_num: int, path: str):
            resolved = self._resolve_json_ref(step_results.get(step_num), path)
            if resolved is not None:
                return resolved
            if step_num in blocked_steps:
                return None
            return self._resolve_json_ref(fallback_step_results.get(step_num), path)

        if isinstance(value, str):
            full = self._STEP_REF_RE.fullmatch(value.strip())
            if full:
                resolved = _lookup(int(full.group(1)), full.group(2))
                return resolved if resolved is not None else value

            def _sub(m):
                resolved = _lookup(int(m.group(1)), m.group(2))
                return str(resolved) if resolved is not None else m.group(0)

            return self._STEP_REF_RE.sub(_sub, value)
        if isinstance(value, dict):
            return {
                k: self._resolve_step_refs(
                    v,
                    step_results,
                    fallback_step_results=fallback_step_results,
                    blocked_steps=blocked_steps,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                self._resolve_step_refs(
                    item,
                    step_results,
                    fallback_step_results=fallback_step_results,
                    blocked_steps=blocked_steps,
                )
                for item in value
            ]
        return value

    def _replace_runtime_placeholders(self, value):
        """Replace generic runtime placeholders with concrete sample values."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in _VIDEO_PATH_PLACEHOLDERS:
                resolved_video_path = str(getattr(self, "video_path", "") or "").strip()
                return resolved_video_path or value
            return value
        if isinstance(value, dict):
            return {k: self._replace_runtime_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._replace_runtime_placeholders(item) for item in value]
        return value

    def _collect_unresolved_step_refs(self, value, found=None):
        if found is None:
            found = []
        if isinstance(value, str):
            matches = _UNRESOLVED_STEP_REF_RE.findall(value)
            for match in matches:
                if match not in found:
                    found.append(match)
            return found
        if isinstance(value, dict):
            for item in value.values():
                self._collect_unresolved_step_refs(item, found)
            return found
        if isinstance(value, list):
            for item in value:
                self._collect_unresolved_step_refs(item, found)
            return found
        return found

    def _dependency_blocked_result(
        self,
        tool_name: str,
        reason: str,
        *,
        blocked_steps: Optional[List[int]] = None,
        unresolved_refs: Optional[List[str]] = None,
    ) -> dict:
        result = {
            "ok": False,
            "error": f"{tool_name} blocked: {reason}",
            "blocked_by_dependency": True,
        }
        if blocked_steps:
            result["blocked_steps"] = [int(step) for step in blocked_steps]
        if unresolved_refs:
            result["unresolved_references"] = list(unresolved_refs)
        return result

    def _extract_json_payload(self, text):
        return self._extract_json_payload_with_schema(
            text,
            model_cls=Union[Dict[str, Any], List[Any]],
        )

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

    def _extract_top_level_json_values(self, text) -> list:
        if not isinstance(text, str):
            return []

        values = []
        decoder = json.JSONDecoder()
        length = len(text)
        idx = 0
        while idx < length:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break

            next_starts = [pos for pos in (text.find("{", idx), text.find("[", idx)) if pos != -1]
            if not next_starts:
                break
            idx = min(next_starts)

            try:
                value, end = decoder.raw_decode(text[idx:])
            except Exception:
                idx += 1
                continue

            values.append(value)
            idx += max(end, 1)

        return values

    def _validate_json_payload(self, payload, model_cls=None):
        if model_cls is None:
            return payload
        try:
            if hasattr(model_cls, "parse_obj"):
                validated = model_cls.parse_obj(payload)
                if isinstance(validated, BaseModel):
                    root_value = getattr(validated, "__root__", None)
                    if root_value is not None:
                        return root_value
                    return validated.dict(exclude_none=False)
                return validated
            validated = parse_obj_as(model_cls, payload)
            if isinstance(validated, BaseModel):
                root_value = getattr(validated, "__root__", None)
                if root_value is not None:
                    return root_value
                return validated.dict(exclude_none=False)
            return validated
        except (ValidationError, TypeError, ValueError):
            return None

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

        for value in self._extract_top_level_json_values(text):
            validated = self._validate_json_payload(value, model_cls)
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

    def _extract_generator_payload(self, text):
        return self._extract_json_payload_with_schema(
            text,
            model_cls=TraceGeneratorOutputModel,
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

    def _segment_dense_captions_cache_path(self, segment_size_s: float) -> Path:
        video_id = Path(str(self.video_path)).stem
        return Path(self.dataset_folder) / "dense_captions" / video_id / f"segment_captions_{float(segment_size_s)}s.json"

    def _format_dense_caption_evidence(self, dense: Any) -> str:
        if not isinstance(dense, dict):
            return str(dense)[:4000]
        parts: List[str] = []
        ov = dense.get("overall_summary")
        if ov:
            parts.append(f"Overall summary: {ov}")
        for cap in dense.get("captions") or []:
            if not isinstance(cap, dict):
                continue
            parts.append(
                f"Span {cap.get('start')}–{cap.get('end')}: "
                f"visual={cap.get('visual', '')}; audio={cap.get('audio', '')}; "
                f"on_screen_text={cap.get('on_screen_text', '')}; "
                f"actions={cap.get('actions', [])}; objects={cap.get('objects', [])}"
            )
        return "\n".join(parts) if parts else json.dumps(dense, ensure_ascii=False)[:4000]

    def _caption_summary_fallback(self, dense: Any) -> str:
        if not isinstance(dense, dict):
            return ""
        visuals: List[str] = []
        for cap in dense.get("captions") or []:
            if isinstance(cap, dict) and cap.get("visual"):
                visuals.append(str(cap["visual"]).strip())
        text = " ".join(visuals).strip() or str(dense.get("overall_summary") or "").strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return " ".join(sentences[:5]).strip()

    def _summarize_segment_caption_object(self, dense: dict, start: float, end: float) -> str:
        evidence = self._format_dense_caption_evidence(dense)
        prompt = (
            f"You are summarizing a video segment from {start:.1f}s to {end:.1f}s.\n\n"
            f"Dense captioner output:\n{evidence}\n\n"
            "Write a 3–5 sentence summary of what is visually and audibly happening in this segment. "
            "Stick strictly to the evidence above; do not invent details. Output plain text only, no JSON."
        )
        try:
            summarize = getattr(self, "_vlm_summarize_text", None)
            if callable(summarize):
                text = (summarize(prompt) or "").strip()
                if text and len(text) > 20:
                    return text
        except Exception as ex:
            print(f"  [_summarize_segment_caption_object] {ex}")
        return self._caption_summary_fallback(dense)

    def _build_segment_dense_captions(self, segment_size_s: float = 30.0) -> None:
        """Populate self._segment_captions_cache; uses disk cache when available."""
        segment_size_s = float(segment_size_s)
        self._segment_captions_cache = []
        cache_path = self._segment_dense_captions_cache_path(segment_size_s)
        if cache_path.is_file():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    abs(float(raw.get("segment_size_s", 0)) - segment_size_s) < 1e-6
                    and str(raw.get("video_path", "")) == str(self.video_path)
                ):
                    self._segment_captions_cache = list(raw.get("segments") or [])
                    print(f"  Loaded segment captions cache → {cache_path}")
                    return
            except Exception as ex:
                print(f"  segment captions cache read failed: {ex}")

        segments: List[dict] = []
        dur = float(self.duration or 0.0)
        t = 0.0
        while t < dur:
            end = min(t + segment_size_s, dur)
            if end - t < 0.5:
                break
            print(f"  [segment dense captions] {t:.1f}s – {end:.1f}s ...")
            dense = self._run_dense_captioner_interval(
                t, end, granularity="segment", focus_query=""
            )
            if not isinstance(dense, dict):
                dense = {"captions": [], "captioned_range": {"start": t, "end": end}}
            summary = self._summarize_segment_caption_object(dense, t, end)
            segments.append(
                {
                    "start": float(t),
                    "end": float(end),
                    "dense_caption": dense,
                    "caption_summary": summary,
                }
            )
            t += segment_size_s

        self._segment_captions_cache = segments
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "segment_size_s": segment_size_s,
                        "video_path": str(self.video_path),
                        "segments": segments,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  Saved segment captions cache → {cache_path}")
        except Exception as ex:
            print(f"  segment captions cache write failed: {ex}")

    def _get_video_caption_summary(self) -> str:
        summary = str(getattr(self, "_video_caption_summary", "") or "").strip()
        if summary:
            return summary

        cache = list(getattr(self, "_segment_captions_cache", None) or [])
        if not cache:
            try:
                self._build_segment_dense_captions(getattr(self, "segment_size_s", 30.0) or 30.0)
            except Exception as ex:
                print(f"  [_get_video_caption_summary] segment caption build failed: {ex}")
            cache = list(getattr(self, "_segment_captions_cache", None) or [])

        segment_summaries: List[str] = []
        for seg in cache:
            text = str(seg.get("caption_summary", "") or "").strip()
            if not text:
                text = self._caption_summary_fallback(seg.get("dense_caption"))
            if not text:
                continue
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) or 0.0)
            segment_summaries.append(f"{start:.1f}s-{end:.1f}s: {text}")

        if not segment_summaries:
            self._video_caption_summary = ""
            return ""

        joined = "\n".join(segment_summaries)
        prompt = (
            "You are summarizing an entire video from caption summaries.\n\n"
            f"Segment caption summaries:\n{joined}\n\n"
            "Write a concise 4-6 sentence summary of the complete video. Mention the "
            "main setting, subjects, event progression, and major phase changes. "
            "Use only the evidence in the segment summaries. Output plain text only."
        )
        try:
            summarize = getattr(self, "_vlm_summarize_text", None)
            if callable(summarize):
                summary = (summarize(prompt) or "").strip()
        except Exception as ex:
            print(f"  [_get_video_caption_summary] summarize failed: {ex}")

        if not summary:
            fallback_text = " ".join(segment_summaries)
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", fallback_text) if s.strip()]
            summary = " ".join(sentences[:8]).strip()

        self._video_caption_summary = summary
        return summary

    def _build_segment_index(self, segment_size_s: float = 30.0) -> None:
        """Join cached segment captions with frame-embedding centroids and ASR snippets."""
        segment_size_s = float(segment_size_s)
        self._segment_index = []
        cache = list(getattr(self, "_segment_captions_cache", None) or [])
        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        frame_embs: Optional[torch.Tensor] = None
        paths_list: List[str] = []
        path_to_idx: Dict[str, int] = {}
        if emb_path.is_file() and paths_path.is_file():
            try:
                paths_list = json.loads(paths_path.read_text(encoding="utf-8"))
                frame_embs = torch.load(emb_path, map_location="cpu").float()
                path_to_idx = {p: i for i, p in enumerate(paths_list)}
            except Exception as ex:
                print(f"  [_build_segment_index] embedding load failed: {ex}")
                frame_embs = None
                paths_list = []
                path_to_idx = {}

        for seg in cache:
            s, e = float(seg["start"]), float(seg["end"])
            asr = self._get_asr_result_from_subtitles(s, e).get("full_transcript", "") or ""
            centroid: Optional[torch.Tensor] = None
            if frame_embs is not None and paths_list:
                idxs: List[int] = []
                for p in paths_list:
                    try:
                        ts = float(self._timestamp_from_dense_frame_path(p))
                    except Exception:
                        continue
                    if s <= ts < e:
                        j = path_to_idx.get(p)
                        if j is not None and j < frame_embs.shape[0]:
                            idxs.append(j)
                if idxs:
                    chunk = frame_embs[idxs].float()
                    chunk = chunk / chunk.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    c = chunk.mean(dim=0)
                    centroid = c / c.norm(p=2, dim=-1).clamp(min=1e-8)

            self._segment_index.append(
                {
                    "start": s,
                    "end": e,
                    "dense_caption": seg.get("dense_caption"),
                    "caption_summary": seg.get("caption_summary", ""),
                    "asr_snippet": asr.strip(),
                    "centroid_emb": centroid,
                }
            )

    def _retrieve_relevant_segments(self, question: str, top_k: int = 5) -> List[dict]:
        q = (question or "").strip()
        if not q:
            return []
        top_k = max(1, int(top_k))
        index = list(getattr(self, "_segment_index", None) or [])
        with_centroid = [s for s in index if s.get("centroid_emb") is not None]
        if not with_centroid:
            return []

        def _as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float()
            return torch.tensor(x, dtype=torch.float32)

        try:
            embedder = self._get_or_load_frame_embedder()
            with self._frame_embedder_inference_context():
                q_emb = _as_tensor(
                    embedder.process(
                        [
                            {
                                "text": q,
                                "instruction": "Retrieve frames relevant to the user's query.",
                            }
                        ]
                    )
                )
            if q_emb.dim() > 1:
                q_emb = q_emb[0]
            q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            qv = q_emb.float().flatten()

            scored: List[tuple] = []
            for seg in with_centroid:
                c = seg["centroid_emb"]
                if not isinstance(c, torch.Tensor):
                    continue
                cv = c.float().flatten()
                sim = float(torch.dot(qv, cv))
                scored.append((sim, seg))
            scored.sort(key=lambda x: -x[0])
            out: List[dict] = []
            for sim, seg in scored[:top_k]:
                dc = seg.get("dense_caption")
                out.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "relevance_score": round(sim, 6),
                        "dense_caption": dc,
                        "asr_snippet": seg.get("asr_snippet", ""),
                    }
                )
            return out
        except Exception as ex:
            print(f"  [_retrieve_relevant_segments] {ex}")
            return []

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
        out: Dict[str, Any] = {
            "asr_transcript": asr_result.get("full_transcript", ""),
            "dense_captions": None,
            "audio_events": None,
            "keyframe_index": [],
            "video_overview": [],
        }
        for seg in getattr(self, "_segment_index", None) or []:
            out["video_overview"].append(
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "caption_summary": seg.get("caption_summary", ""),
                    "asr_snippet": seg.get("asr_snippet", ""),
                }
            )
        if getattr(self, "use_retrieved_context", False):
            k = int(getattr(self, "retrieval_top_k", 5) or 5)
            out["retrieved_context"] = self._retrieve_relevant_segments(
                getattr(self, "question", "") or "", top_k=k
            )
        return out

    def _extract_final_answer(self, text: str) -> str:
        try:
            answer_content = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)[-1].strip()
            answer_content = re.sub(r"\s+", " ", answer_content)
            return answer_content if answer_content else "-"
        except Exception:
            return "-"

    def _normalize_answer(self, answer: str) -> str:
        return re.sub(r"\s+", " ", str(answer)).strip().lower()

    def _resolve_mcq_answer(self, answer: str, options: list = None) -> str:
        """If *answer* is a bare MCQ letter (e.g. "A", "(B)", "C."), expand it to
        the full option text so that downstream comparisons work correctly."""
        ans = (answer or "").strip()
        opts = list(options if options is not None else (self.options or []))
        if not opts or not ans:
            return ans
        letter_match = re.match(r'^\(?([A-Za-z])\)?\.?$', ans)
        if not letter_match:
            return ans
        letter = letter_match.group(1).upper()
        for opt in opts:
            opt_str = str(opt).strip()
            if re.match(rf'^\(?{letter}[.):\s]', opt_str, re.IGNORECASE):
                # Strip the leading "A. " / "A) " / "(A) " prefix and return the text
                text_part = re.sub(r'^\(?[A-Za-z][.):\s]+', '', opt_str).strip()
                return text_part if text_part else opt_str
        return ans  # no matching option found — return as-is

    def _answers_match(self, predicted: str, gold: str, options: list = None) -> bool:
        """Normalised equality check.  MCQ letters in *predicted* are first
        resolved to full option text so "A" matches "The man is walking"."""
        resolved = self._resolve_mcq_answer(predicted, options)
        return self._normalize_answer(resolved) == self._normalize_answer(gold)

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

    def _trace_has_unresolved_conclusion(self, trace_steps: list, unresolved_issues: list | None = None) -> bool:
        issues = [str(item).strip() for item in (unresolved_issues or []) if str(item).strip()]
        if issues:
            return True

        joined = "\n".join(str(step).strip() for step in (trace_steps or []) if str(step).strip())
        if not joined:
            return False

        for pattern in self._UNRESOLVED_TRACE_PATTERNS:
            if pattern.search(joined):
                return True
        return False
