import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import refiner_debug
from refine_prompt import (
    planner_prompt,
    refiner_prompt,
    self_contained_trace_prompt,
    trace_generator_prompt,
    verifier_prompt,
)


class RefinerAgentsMixin:
    _NON_STANDALONE_TRACE_PATTERNS = [
        re.compile(r"\boriginal trace\b", flags=re.IGNORECASE),
        re.compile(r"\boriginal answer\b", flags=re.IGNORECASE),
        re.compile(r"\bprevious answer\b", flags=re.IGNORECASE),
        re.compile(r"\bprevious jump\b", flags=re.IGNORECASE),
        re.compile(r"\bearlier supported\b", flags=re.IGNORECASE),
        re.compile(r"\bearlier supported evidence\b", flags=re.IGNORECASE),
        re.compile(r"\battempted repair\b", flags=re.IGNORECASE),
        re.compile(r"\btool_outputs\b", flags=re.IGNORECASE),
        re.compile(r"\bchanges_made\b", flags=re.IGNORECASE),
        re.compile(r"\bunresolved_issues\b", flags=re.IGNORECASE),
        re.compile(r"\bplanner\b", flags=re.IGNORECASE),
        re.compile(r"\bverifier\b", flags=re.IGNORECASE),
        re.compile(r"\brefiner\b", flags=re.IGNORECASE),
        re.compile(
            r"\b(?:temporal_grounder|frame_retriever|asr|audio_grounder|ocr|"
            r"spatial_grounder|counter|dense_captioner|action_recognizer|chart_analyzer)"
            r"\s+Step\s+\d+\b",
            flags=re.IGNORECASE,
        ),
    ]

    def _retry_malformed_json_response(
        self,
        raw_output: str,
        model_name: str,
        api_base: list,
        api_keys: list,
        schema_name: str,
        required_keys: list | None = None,
    ) -> str:
        raw_output = str(raw_output or "").strip()
        if not raw_output:
            return ""

        key_text = ", ".join(required_keys or [])
        prompt = (
            f"The following {schema_name} response was malformed or truncated.\n"
            "Repair it into a single valid JSON object.\n"
            "Return ONLY JSON with no markdown or explanation.\n"
        )
        if key_text:
            prompt += f"Required top-level keys: {key_text}.\n"
        prompt += "\nMALFORMED_JSON:\n```json\n" + raw_output + "\n```"
        return self._text2text(
            [{"role": "user", "content": prompt}],
            model_name,
            api_base,
            api_keys,
        )

    def _parse_tool_result_json(self, output_str: str):
        if not output_str or "are:\n" not in output_str:
            return None
        tail = output_str.split("are:\n", 1)[-1].strip()
        parsed = self._extract_json_payload_with_schema(tail, model_cls=Dict[str, Any])
        return parsed if isinstance(parsed, dict) else None

    def _sanitize_tool_output_for_refiner(self, tool_name: str, output_str: str) -> str:
        return output_str

    def _extract_tool_confidence(self, tool_name: str, result: dict) -> float:
        if not isinstance(result, dict):
            return 0.0
        if tool_name == "temporal_grounder":
            segs = result.get("segments") or []
            if not segs:
                return 0.0
            return max(float(s.get("confidence", 0) or 0) for s in segs)
        if tool_name == "ocr":
            dets = result.get("detections") or []
            if not dets:
                return 0.0
            confs = [float(d.get("confidence", 0) or 0) for d in dets]
            return sum(confs) / len(confs) if confs else 0.0
        if tool_name == "spatial_grounder":
            dets = result.get("detections") or []
            if not dets and isinstance(result.get("frames"), list):
                frame_scores = []
                for item in result.get("frames") or []:
                    if not isinstance(item, dict):
                        continue
                    fdets = item.get("detections") or []
                    if isinstance(fdets, list) and fdets:
                        frame_scores.append(
                            max(float(d.get("confidence", 0) or 0) for d in fdets if isinstance(d, dict))
                        )
                if frame_scores:
                    return max(frame_scores)
            if not dets:
                return 0.0
            return max(float(d.get("confidence", 0) or 0) for d in dets)
        if tool_name == "counter":
            if (result.get("confidence") in {None, 0, 0.0}) and isinstance(result.get("frames"), list):
                frame_confs = [
                    float(item.get("confidence", 0) or 0)
                    for item in (result.get("frames") or [])
                    if isinstance(item, dict)
                ]
                if frame_confs:
                    return max(frame_confs)
            return float(result.get("confidence", 0) or 0)
        # if tool_name == "video_qa_reanswerer":
        #     return float(result.get("confidence", 0) or 0)
        if tool_name == "dense_captioner":
            caps = result.get("captions") or []
            return 1.0 if caps else 0.0
        if tool_name == "action_recognizer":
            acts = result.get("actions") or []
            if not acts:
                return 0.0
            return max(float(a.get("confidence", 0) or 0) for a in acts)
        if tool_name == "asr":
            txt = (result.get("full_transcript") or result.get("transcript") or "").strip()
            if not txt:
                return 0.0
            segs = result.get("segments") or []
            if segs:
                confs = [float(s.get("confidence", 1.0) or 1.0) for s in segs]
                return sum(confs) / len(confs) if confs else 1.0
            return 1.0
        if tool_name == "frame_retriever":
            frames = result.get("frames") or []
            if not frames:
                return 0.0
            return max(float(f.get("relevance_score", 1.0) or 1.0) for f in frames)
        if tool_name == "audio_grounder":
            ev = result.get("events") or []
            if ev:
                return max(float(e.get("confidence", 0) or 0) for e in ev)
            groups = result.get("distinct_event_groups") or []
            if groups:
                return max(float(g.get("confidence", 0.0) or 0.0) for g in groups)
            status = str(result.get("audio_status", "") or "").strip().lower()
            if status in {"ok", "no_match", "analyzed_no_match", "no_subtitle_tags", "empty_window", "too_short"}:
                return 0.1
            return 0.0
        if tool_name == "chart_analyzer":
            score = 0.0
            if str(result.get("chart_type", "") or "").strip() not in {"", "unknown", "other"}:
                score += 0.35
            if result.get("series"):
                score += 0.4
            if result.get("key_observations"):
                score += 0.15
            if str(result.get("query_response", "") or "").strip():
                score += 0.1
            return min(score, 1.0)
        return 0.0

    def _tool_signal_summary(self, tool_name: str, result) -> str:
        if not isinstance(result, (dict, list)):
            return ""

        def _floats(values) -> list[float]:
            out = []
            for value in values:
                if value is None:
                    continue
                try:
                    out.append(float(value))
                except Exception:
                    continue
            return out

        if tool_name == "temporal_grounder" and isinstance(result, dict):
            confs = _floats(
                s.get("confidence")
                for s in (result.get("segments") or [])
                if isinstance(s, dict) and s.get("confidence") is not None
            )
            return f"top segment confidence {max(confs):.4f}" if confs else ""

        if tool_name == "frame_retriever" and isinstance(result, dict):
            scores = _floats(
                f.get("relevance_score")
                for f in (result.get("frames") or [])
                if isinstance(f, dict) and f.get("relevance_score") is not None
            )
            return f"top relevance_score {max(scores):.4f}" if scores else ""

        if tool_name == "ocr" and isinstance(result, dict):
            confs = _floats(
                d.get("confidence")
                for d in (result.get("detections") or [])
                if isinstance(d, dict) and d.get("confidence") is not None
            )
            return f"avg detection confidence {sum(confs) / len(confs):.4f}" if confs else ""

        if tool_name == "spatial_grounder" and isinstance(result, dict):
            confs = _floats(
                d.get("confidence")
                for d in (result.get("detections") or [])
                if isinstance(d, dict) and d.get("confidence") is not None
            )
            if confs:
                return f"max detection confidence {max(confs):.4f}"
            frame_scores = []
            for item in result.get("frames") or []:
                if not isinstance(item, dict):
                    continue
                fdets = item.get("detections") or []
                frame_scores.extend(
                    _floats(
                        d.get("confidence")
                        for d in fdets
                        if isinstance(d, dict) and d.get("confidence") is not None
                    )
                )
            return f"max frame detection confidence {max(frame_scores):.4f}" if frame_scores else ""

        if tool_name == "counter" and isinstance(result, dict):
            top_conf = result.get("confidence")
            if top_conf is not None:
                try:
                    return f"confidence {float(top_conf):.4f}"
                except Exception:
                    pass
            frame_scores = _floats(
                item.get("confidence")
                for item in (result.get("frames") or [])
                if isinstance(item, dict) and item.get("confidence") is not None
            )
            return f"max frame confidence {max(frame_scores):.4f}" if frame_scores else ""

        if tool_name == "asr" and isinstance(result, dict):
            segs = [
                s for s in (result.get("segments") or []) if isinstance(s, dict) and s.get("confidence") is not None
            ]
            confs = _floats(s.get("confidence") for s in segs)
            return f"avg segment confidence {sum(confs) / len(confs):.4f}" if confs else ""

        if tool_name == "audio_grounder" and isinstance(result, dict):
            event_confs = _floats(
                e.get("confidence")
                for e in (result.get("events") or [])
                if isinstance(e, dict) and e.get("confidence") is not None
            )
            if event_confs:
                return f"top event confidence {max(event_confs):.4f}"
            group_confs = _floats(
                g.get("confidence")
                for g in (result.get("distinct_event_groups") or [])
                if isinstance(g, dict) and g.get("confidence") is not None
            )
            return f"top group confidence {max(group_confs):.4f}" if group_confs else ""

        if tool_name == "action_recognizer":
            confs = _floats(
                a.get("confidence")
                for a in (result.get("actions") or [])
                if isinstance(a, dict) and a.get("confidence") is not None
            )
            if not confs and isinstance(result, list):
                confs = _floats(
                    a.get("confidence")
                    for a in result
                    if isinstance(a, dict) and a.get("confidence") is not None
                )
            return f"top action confidence {max(confs):.4f}" if confs else ""

        return ""

    def _tool_signal_summary_from_output(self, tool_name: str, output_text: str) -> str:
        parsed = self._parse_tool_result_json(output_text)
        if tool_name == "action_recognizer" and parsed is None:
            tail = str(output_text or "").split("are:\n", 1)[-1].strip()
            try:
                parsed = json.loads(tail)
            except Exception:
                parsed = None
        return self._tool_signal_summary(tool_name, parsed or {})

    def _compact_iteration_summary(
        self, iteration_idx: int, verifier_output, refiner_output, executed_tools: list
    ) -> dict:
        tools_executed = []
        threshold = float(os.getenv("REFINER_TOOL_RESOLVED_THRESHOLD", "0.7"))
        for item in executed_tools or []:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool", "")
            purpose = item.get("purpose", "")
            parsed = self._parse_tool_result_json(item.get("output", ""))
            conf = self._extract_tool_confidence(tool, parsed or {})
            tools_executed.append(
                {
                    "tool": tool,
                    "purpose": purpose,
                    "confidence": round(conf, 4),
                    "resolved": bool(conf >= threshold),
                }
            )

        errors_fixed = []
        errors_remaining = []
        if isinstance(refiner_output, dict):
            for ch in refiner_output.get("changes_made") or []:
                if isinstance(ch, dict):
                    op = ch.get("operation", "")
                    si = ch.get("step_index")
                    errors_fixed.append(f"{op} step {si}")
            errors_remaining = list(refiner_output.get("unresolved_issues") or [])

        v_out = verifier_output if isinstance(verifier_output, dict) else {}
        return {
            "iteration": iteration_idx + 1,
            "verifier_verdict": v_out.get("verdict"),
            "tools_executed": tools_executed,
            "errors_fixed": errors_fixed[:20],
            "errors_remaining": errors_remaining[:20],
            "answer_changed": bool(refiner_output.get("answer_changed"))
            if isinstance(refiner_output, dict)
            else False,
        }

    def _compact_generation_summary(self, generation_record: dict) -> dict:
        if not isinstance(generation_record, dict):
            return {
                "iteration": 0,
                "phase": "initial_trace_generation",
                "verifier_verdict": None,
                "tools_executed": [],
                "errors_fixed": [],
                "errors_remaining": [],
                "answer_changed": False,
            }

        tools_executed = []
        threshold = float(os.getenv("REFINER_TOOL_RESOLVED_THRESHOLD", "0.7"))
        for item in generation_record.get("executed_tools") or []:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool", "")
            purpose = item.get("purpose", "")
            parsed = self._parse_tool_result_json(item.get("output", ""))
            conf = self._extract_tool_confidence(tool, parsed or {})
            tools_executed.append(
                {
                    "tool": tool,
                    "purpose": purpose,
                    "confidence": round(conf, 4),
                    "resolved": bool(conf >= threshold),
                }
            )

        errors_fixed = []
        errors_remaining = []
        refiner_output = generation_record.get("refiner_output")
        if isinstance(refiner_output, dict):
            for ch in refiner_output.get("changes_made") or []:
                if isinstance(ch, dict):
                    op = ch.get("operation", "")
                    si = ch.get("step_index")
                    errors_fixed.append(f"{op} step {si}")
            errors_remaining = list(refiner_output.get("unresolved_issues") or [])

        diagnosis = generation_record.get("diagnosis")
        diag = diagnosis if isinstance(diagnosis, dict) else {}
        return {
            "iteration": 0,
            "phase": "initial_trace_generation",
            "verifier_verdict": diag.get("verdict"),
            "tools_executed": tools_executed,
            "errors_fixed": errors_fixed[:20],
            "errors_remaining": errors_remaining[:20],
            "answer_changed": bool(refiner_output.get("answer_changed"))
            if isinstance(refiner_output, dict)
            else False,
        }

    def _iteration_context_block(self, iteration: int, max_iterations: int, history: list) -> str:
        hist = history or []
        return (
            f"ITERATION: {iteration + 1}/{max_iterations}\n"
            f"PREVIOUS_ITERATIONS_SUMMARY:\n{json.dumps(hist, ensure_ascii=False, indent=2)}\n"
        )

    def _build_verifier_prompt(
        self,
        trace_steps: list,
        trace_answer: str,
        question_text: str = None,
        iteration: int = 0,
        history: list = None,
        max_iterations: int = 3,
    ) -> str:
        ctx = self._iteration_context_block(iteration, max_iterations, history or [])
        question_block = question_text if question_text is not None else self._format_question_with_options()
        return (
            ctx
            + "\n"
            + verifier_prompt.strip()
            + "\n\nQUESTION:\n"
            + question_block
            + "\n\nTRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nTEXT_ONLY_MODE:\n"
            + "This verifier call has no access to video, audio, frames, OCR outputs, or hidden tool state. Use only the text in this prompt and the iteration summary above. There is intentionally no separate answer field in this verifier call; infer any final conclusion only from the trace itself. Treat unsupported sensory claims as unsupported rather than observed. However, if the TRACE explicitly attributes a claim to a named tool result and phrases it as a reported tool output, treat that attribution as textual evidence rather than as an unsupported direct observation. Tool names in the iteration summary alone do not supply missing numeric values. Set error_categories[].evidence to null or \"N/A (text-only pass)\".\n"
        )

    def _build_verifier_l2_prompt(self, trace_steps: list, trace_answer: str, l1_diagnosis: dict) -> str:
        diag = json.dumps(l1_diagnosis, ensure_ascii=False, indent=2) if isinstance(l1_diagnosis, dict) else str(
            l1_diagnosis
        )
        return (
            "You are the Level 2 factual verifier. You see sampled frames from the video (as a video input).\n"
            "Using ONLY what you observe in the video, confirm or revise the Level-1 diagnosis.\n"
            "Respond with JSON ONLY using the same schema as the Verifier:\n"
            '{ "verdict": "PASS" or "FAIL", "answer_correct": true/false, '
            '"trace_quality_scores": { "perceptual_correctness": 0-10, "temporal_accuracy": 0-10, '
            '"logical_coherence": 0-10, "completeness": 0-10 }, '
            '"error_categories": [...], "confidence": 0.0-1.0, "summary": "..." }\n'
            "Populate error_categories[].evidence with concrete visual/audio observations.\n\n"
            "LEVEL_1_DIAGNOSIS:\n"
            + diag
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nTRACE:\n"
            + self._format_trace_steps(trace_steps)
        )

    def _should_skip_l2(self, history: list, l1_out: dict) -> bool:
        if not isinstance(l1_out, dict) or l1_out.get("verdict") != "PASS":
            return False
        if not history:
            return False
        last = history[-1]
        tools = last.get("tools_executed") or []
        if not tools:
            return False
        return all(bool(t.get("resolved")) for t in tools)

    def _merge_verifier_l1_l2(self, l1: dict, l2: dict) -> dict:
        if not isinstance(l1, dict):
            return l2 if isinstance(l2, dict) else {}
        if not isinstance(l2, dict):
            return dict(l1)
        out = dict(l1)
        v1, v2 = l1.get("verdict"), l2.get("verdict")
        out["verdict"] = "PASS" if (v1 == "PASS" and v2 == "PASS") else "FAIL"
        errs = []
        for e in l1.get("error_categories") or []:
            if isinstance(e, dict):
                errs.append(dict(e))
        for e in l2.get("error_categories") or []:
            if isinstance(e, dict):
                errs.append(dict(e))
        out["error_categories"] = errs
        out["confidence"] = min(
            float(l1.get("confidence", 0) or 0),
            float(l2.get("confidence", 0) or 0),
        )
        out["summary"] = (
            f"L1: {l1.get('summary', '')} | L2: {l2.get('summary', '')}"
        ).strip(" |")
        if "answer_correct" in l2:
            out["answer_correct"] = l2.get("answer_correct") and l1.get("answer_correct", True)
        if isinstance(l2.get("trace_quality_scores"), dict):
            ts = dict(l1.get("trace_quality_scores") or {})
            for k, v in l2["trace_quality_scores"].items():
                if k in ts:
                    ts[k] = min(float(ts[k]), float(v))
                else:
                    ts[k] = float(v)
            out["trace_quality_scores"] = ts
        return out

    def _call_verifier(
        self,
        trace_steps: list,
        trace_answer: str,
        question_text: str = None,
        iteration: int = 0,
        history: list = None,
        max_iterations: int = 3,
    ):
        ibase = getattr(self, "_refinement_debug_iter_dir", None)
        v_out_dir = None
        if ibase:
            v_out_dir = refiner_debug.ensure_outputs_dir(Path(ibase) / "verifier")

        prompt = self._build_verifier_prompt(
            trace_steps,
            trace_answer,
            question_text=question_text,
            iteration=iteration,
            history=history,
            max_iterations=max_iterations,
        )
        messages = [{"role": "user", "content": prompt}]
        if v_out_dir:
            refiner_debug.write_json(
                v_out_dir,
                "l1_model_input.json",
                {"model": self.verifier_model_name, "input": messages},
            )

        l1_raw = self._text2text(
            messages, self.verifier_model_name, self.verifier_api_base, self.verifier_api_keys
        )
        if v_out_dir:
            refiner_debug.write_text(v_out_dir, "l1_output.txt", l1_raw or "")

        l1_parsed = self._extract_verifier_payload(l1_raw)
        l1_out = l1_parsed if isinstance(l1_parsed, dict) else None

        def _write_l1_only_merged():
            if not v_out_dir:
                return
            merged_local = l1_out if isinstance(l1_out, dict) else {}
            refiner_debug.write_json(v_out_dir, "merged_verdict.json", merged_local)
            refiner_debug.write_text(v_out_dir, "combined_raw.txt", l1_raw or "")

        skip_l2 = self._should_skip_l2(history or [], l1_out or {})
        if skip_l2:
            _write_l1_only_merged()
            return l1_raw, l1_out

        # L2 (video) verifier: disabled for now. Re-enable by uncommenting below and
        # removing the _write_l1_only_merged / return block that follows.
        _write_l1_only_merged()
        return l1_raw, l1_out

        # frame_paths, timestamps, _, _ = self._get_frames_for_range(None, None, fps=0.5)
        # if not frame_paths:
        #     _write_l1_only_merged()
        #     return l1_raw, l1_out
        #
        # l2_default = {
        #     "verdict": "FAIL",
        #     "answer_correct": False,
        #     "trace_quality_scores": {
        #         "perceptual_correctness": 0,
        #         "temporal_accuracy": 0,
        #         "logical_coherence": 10,
        #         "completeness": 5,
        #     },
        #     "error_categories": [],
        #     "confidence": 0.0,
        #     "summary": "L2 skipped: no frames",
        # }
        # l2_prompt = self._build_verifier_l2_prompt(trace_steps, trace_answer, l1_out or {})
        # if v_out_dir:
        #     self._refinement_debug_vlm_outputs_dir = v_out_dir
        #     self._refinement_debug_vlm_input_basename = "l2_model_input.json"
        # try:
        #     l2_parsed = self._run_vlm_json(l2_prompt, frame_paths, timestamps, l2_default)
        # finally:
        #     self._refinement_debug_vlm_outputs_dir = None
        #     self._refinement_debug_vlm_input_basename = None
        #
        # l2_out = l2_parsed if isinstance(l2_parsed, dict) else l2_default
        #
        # merged = self._merge_verifier_l1_l2(l1_out or {}, l2_out)
        # merged_raw = json.dumps(merged, ensure_ascii=False, indent=2)
        # combined_raw = f"=== L1 (text) ===\n{l1_raw}\n\n=== L2 (video) ===\n{json.dumps(l2_out, ensure_ascii=False, indent=2)}\n\n=== MERGED ===\n{merged_raw}"
        # if v_out_dir:
        #     refiner_debug.write_json(v_out_dir, "l2_output.json", l2_out)
        #     refiner_debug.write_json(v_out_dir, "merged_verdict.json", merged)
        #     refiner_debug.write_text(v_out_dir, "combined_raw.txt", combined_raw)
        #
        # return combined_raw, merged

    def _build_planner_prompt(
        self,
        trace_steps: list,
        trace_answer: str,
        diagnosis,
        iteration: int = 0,
        history: list = None,
        max_iterations: int = 3,
    ) -> str:
        diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, indent=2) if isinstance(diagnosis, dict) else str(
            diagnosis
        )
        # Provide the planner with caption-derived full-video context.
        ctx = self._iteration_context_block(iteration, max_iterations, history or [])
        last_tools = ""
        if history:
            last_tools = json.dumps(history[-1].get("tools_executed", []), ensure_ascii=False, indent=2)
        video_caption_summary = ""
        get_video_caption_summary = getattr(self, "_get_video_caption_summary", None)
        if callable(get_video_caption_summary):
            video_caption_summary = str(get_video_caption_summary() or "").strip()
        return (
            ctx
            + "\n"
            + planner_prompt.strip()
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nTRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nANSWER:\n"
            + trace_answer
            + "\n\nDIAGNOSIS:\n"
            + diagnosis_text
            + "\n\nVIDEO_CAPTION_SUMMARY:\n"
            + (video_caption_summary or "(unavailable)")
            + "\n\nPREVIOUS_TOOL_RESULTS_SUMMARY (last iteration tools + confidence):\n"
            + (last_tools or "[]")
        )

    def _step_depends_on_tool(self, step_num: int, tool_name: str, call_by_step: dict, seen=None) -> bool:
        seen = seen or set()
        if step_num in seen:
            return False
        seen.add(step_num)
        call = call_by_step.get(step_num)
        if not isinstance(call, dict):
            return False
        if str(call.get("tool", "") or "").strip() == tool_name:
            return True
        for dep in call.get("depends_on", []) or []:
            try:
                dep_num = int(dep)
            except (TypeError, ValueError):
                continue
            if self._step_depends_on_tool(dep_num, tool_name, call_by_step, seen):
                return True
        return False

    def _chart_plan_needs_temporal_grounding(
        self,
        frame_call: dict,
        chart_call: dict,
        diagnosis,
        history: list | None = None,
    ) -> bool:
        frame_args = frame_call.get("arguments", {}) if isinstance(frame_call, dict) else {}
        chart_args = chart_call.get("arguments", {}) if isinstance(chart_call, dict) else {}
        frame_query = str(frame_args.get("query", "") or "").strip()
        chart_query = str(chart_args.get("query", "") or "").strip()
        if not frame_query:
            return False
        if frame_args.get("timestamps"):
            return False

        blobs = [
            str(self.question or ""),
            self._format_question_with_options(),
            frame_query,
            chart_query,
        ]
        if diagnosis is not None:
            blobs.append(json.dumps(diagnosis, ensure_ascii=False))
        if history:
            blobs.append(json.dumps(history, ensure_ascii=False))
        text = " ".join(blob for blob in blobs if blob).lower()

        chart_cues = (
            "chart" in text
            or "graph" in text
            or "plot" in text
            or "infographic" in text
            or "dashboard" in text
            or "table" in text
        )
        comparison_cues = any(
            cue in text
            for cue in (
                "difference",
                "discrepancy",
                "compare",
                "comparison",
                "gap",
                "highest",
                "lowest",
                "largest",
                "smallest",
                "versus",
                "between",
            )
        )
        multi_metric_cues = any(
            cue in text
            for cue in (
                "metrics",
                "attributes",
                "series",
                "store cleanliness",
                "value for dollar",
                "availability of items",
            )
        )
        incomplete_chart_cues = any(
            cue in text
            for cue in (
                "does not contain",
                "does not provide",
                "missing",
                "lacks",
                "only provides",
                "only include",
                "cannot be computed",
                "partial",
                "incomplete",
                "wrong phase",
                "wrong frame",
                "not visible",
                "not fully shown",
                "not fully rendered",
                "animated",
                "progressively",
            )
        )
        return bool(chart_cues and (comparison_cues or multi_metric_cues or incomplete_chart_cues))

    def _build_chart_temporal_grounder_query(self, frame_call: dict, chart_call: dict) -> str:
        frame_args = frame_call.get("arguments", {}) if isinstance(frame_call, dict) else {}
        chart_args = chart_call.get("arguments", {}) if isinstance(chart_call, dict) else {}
        frame_query = str(frame_args.get("query", "") or "").strip()
        chart_query = str(chart_args.get("query", "") or "").strip()
        if re.match(r"^(read|identify|determine|extract|compute|interpret)\b", chart_query, flags=re.IGNORECASE):
            base = frame_query or chart_query or str(self.question or "").strip()
        else:
            base = chart_query or frame_query or str(self.question or "").strip()
        if not base:
            return "segment where the relevant chart or infographic appears on screen"

        base = re.sub(r"^\s*frame\s+(showing|with)\s+", "", base, flags=re.IGNORECASE)
        base = base.rstrip(". ")
        if re.search(r"\b(chart|graph|plot|infographic|dashboard|table)\b", base, flags=re.IGNORECASE):
            return f"segment where {base} appears on screen"
        return f"segment where the relevant chart or infographic appears on screen for: {base}"

    def _repair_planner_chart_grounding(
        self,
        planner_output: dict,
        diagnosis,
        history: list | None = None,
    ) -> dict:
        if not isinstance(planner_output, dict):
            return planner_output

        tool_calls = planner_output.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return planner_output

        ordered_calls = sorted(
            [dict(call) for call in tool_calls if isinstance(call, dict)],
            key=lambda call: int(call.get("step", 0) or 0),
        )
        if not ordered_calls:
            return planner_output

        call_by_step = {
            int(call.get("step", 0) or 0): call
            for call in ordered_calls
            if int(call.get("step", 0) or 0)
        }
        inject_before: dict[int, str] = {}

        for call in ordered_calls:
            if str(call.get("tool", "") or "").strip() != "chart_analyzer":
                continue
            for dep in call.get("depends_on", []) or []:
                try:
                    dep_step = int(dep)
                except (TypeError, ValueError):
                    continue
                frame_call = call_by_step.get(dep_step)
                if not isinstance(frame_call, dict):
                    continue
                if str(frame_call.get("tool", "") or "").strip() != "frame_retriever":
                    continue
                frame_args = frame_call.get("arguments", {}) if isinstance(frame_call.get("arguments"), dict) else {}
                if not str(frame_args.get("query", "") or "").strip():
                    continue
                if frame_args.get("timestamps"):
                    continue
                if self._step_depends_on_tool(dep_step, "temporal_grounder", call_by_step):
                    continue
                if not self._chart_plan_needs_temporal_grounding(frame_call, call, diagnosis, history):
                    continue
                inject_before[dep_step] = self._build_chart_temporal_grounder_query(frame_call, call)

        if not inject_before:
            return planner_output

        staged_calls = []
        for call in ordered_calls:
            old_step = int(call.get("step", 0) or 0)
            if old_step in inject_before:
                staged_calls.append(
                    (
                        ("tg", old_step),
                        {
                            "tool": "temporal_grounder",
                            "arguments": {
                                "video_path": (
                                    (call.get("arguments") or {}).get("video_path")
                                    if isinstance(call.get("arguments"), dict)
                                    else None
                                ),
                                "query": inject_before[old_step],
                            },
                            "purpose": (
                                "Localize the chart / infographic interval before frame retrieval so "
                                "downstream chart reading uses temporally grounded frames instead of "
                                "raw query-ranked hits that may reflect a partial or animated state."
                            ),
                            "depends_on": [],
                        },
                    )
                )
            staged_calls.append((old_step, dict(call)))

        new_step_map = {}
        for new_step, (key, _) in enumerate(staged_calls, start=1):
            new_step_map[key] = new_step

        repaired_calls = []
        for key, call in staged_calls:
            old_deps = call.get("depends_on", []) or []
            new_deps = []
            for dep in old_deps:
                try:
                    dep_num = int(dep)
                except (TypeError, ValueError):
                    continue
                mapped = new_step_map.get(dep_num)
                if mapped is not None:
                    new_deps.append(mapped)
            if isinstance(key, int) and key in inject_before:
                new_deps.append(new_step_map[("tg", key)])

            updated_call = dict(call)
            updated_call["step"] = new_step_map[key]
            updated_call["depends_on"] = list(dict.fromkeys(new_deps))
            repaired_calls.append(updated_call)

        repaired = dict(planner_output)
        repaired["tool_calls"] = repaired_calls

        strategy = str(repaired.get("strategy", "") or "").strip()
        strategy_note = (
            "Because the planner does not see the video, this plan first temporally grounds the chart "
            "interval before frame retrieval so animated or partially rendered chart states do not "
            "silently contaminate chart_analyzer."
        )
        if strategy_note not in strategy:
            repaired["strategy"] = strategy_note if not strategy else strategy + " " + strategy_note

        refinstr = str(repaired.get("refinement_instructions", "") or "").strip()
        refinstr_note = (
            "Use the temporally grounded frame bundle as the chart-reading evidence anchor. Do not "
            "treat a raw query-ranked chart frame as guaranteed complete when the task needs full "
            "chart contents, multiple metrics, or cross-metric comparison."
        )
        if refinstr_note not in refinstr:
            repaired["refinement_instructions"] = (
                refinstr_note if not refinstr else refinstr + " " + refinstr_note
            )

        return repaired

    def _call_planner(
        self,
        trace_steps: list,
        trace_answer: str,
        diagnosis,
        iteration: int = 0,
        history: list = None,
        max_iterations: int = 3,
    ):
        ibase = getattr(self, "_refinement_debug_iter_dir", None)
        p_out_dir = None
        if ibase:
            p_out_dir = refiner_debug.ensure_outputs_dir(Path(ibase) / "planner")

        prompt = self._build_planner_prompt(
            trace_steps,
            trace_answer,
            diagnosis,
            iteration=iteration,
            history=history,
            max_iterations=max_iterations,
        )
        messages = [{"role": "user", "content": prompt}]
        if p_out_dir:
            refiner_debug.write_json(
                p_out_dir,
                "model_input.json",
                {"model": self.planner_model_name, "input": messages},
            )

        raw_output = self._text2text(
            messages, self.planner_model_name, self.planner_api_base, self.planner_api_keys
        )
        if p_out_dir:
            refiner_debug.write_text(p_out_dir, "raw_output.txt", raw_output or "")

        parsed_output = self._extract_planner_payload(raw_output)
        repair_raw_output = ""
        if parsed_output is None and str(raw_output or "").strip():
            repair_raw_output = self._retry_malformed_json_response(
                raw_output,
                self.planner_model_name,
                self.planner_api_base,
                self.planner_api_keys,
                schema_name="planner",
                required_keys=["strategy", "tool_calls", "refinement_instructions"],
            )
            if p_out_dir:
                refiner_debug.write_text(p_out_dir, "repair_raw_output.txt", repair_raw_output or "")
            parsed_output = self._extract_planner_payload(repair_raw_output)

        parsed_dict = parsed_output if isinstance(parsed_output, dict) else None
        if parsed_dict is not None:
            parsed_dict = self._repair_planner_chart_grounding(
                parsed_dict,
                diagnosis,
                history or [],
            )
        if p_out_dir and parsed_dict is not None:
            refiner_debug.write_json(p_out_dir, "plan.json", parsed_dict)
        if parsed_dict is None:
            raise RuntimeError("Planner returned malformed JSON and could not be repaired.")

        return raw_output, parsed_dict

    def _normalize_refined_trace(self, raw, fallback: list) -> list:
        def _clean_trace_text(text: str) -> str:
            cleaned = str(text).strip()
            patterns = [
                r"^\d+\.\s*",
                r"^TOOL_OUTPUTS\s+Step\s+\d+\s*[:.\-]?\s*",
                r"^Step\s+\d+\s*[:.\-]?\s*",
            ]
            prev = None
            while cleaned != prev:
                prev = cleaned
                for pattern in patterns:
                    cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
            return cleaned

        if isinstance(raw, list):
            out = []
            for item in raw:
                out.append(_clean_trace_text(str(item)))
            return out
        if isinstance(raw, str):
            lines = [ln.rstrip() for ln in raw.split("\n")]
            out = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                out.append(_clean_trace_text(line))
            return out if out else [raw]
        return list(fallback)

    def _trace_needs_self_contained_rewrite(self, trace_steps) -> bool:
        normalized = self._normalize_refined_trace(trace_steps, [])
        for step in normalized:
            text = str(step or "").strip()
            if not text:
                continue
            for pattern in self._NON_STANDALONE_TRACE_PATTERNS:
                if pattern.search(text):
                    return True
        return False

    def _build_self_contained_trace_prompt(
        self,
        trace_steps: list,
        trace_answer: str,
        executed_tools: list,
    ) -> str:
        tool_lines = []
        for item in executed_tools or []:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool", "")
            safe_output = self._sanitize_tool_output_for_refiner(tool_name, item.get("output", ""))
            tool_lines.append(
                f"Step {item.get('step')}: {tool_name} — {item.get('purpose', '')}\n{safe_output}"
            )
        tools_block = "\n".join(tool_lines) if tool_lines else "(no tool outputs)"
        return (
            self_contained_trace_prompt.strip()
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nCURRENT_REFINED_TRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nCURRENT_REFINED_ANSWER:\n"
            + str(trace_answer or "").strip()
            + "\n\nTOOL_OUTPUTS:\n"
            + tools_block
        )

    def _rewrite_trace_self_contained(
        self,
        trace_steps: list,
        trace_answer: str,
        executed_tools: list,
    ):
        ibase = getattr(self, "_refinement_debug_iter_dir", None)
        out_dir = None
        if ibase:
            out_dir = refiner_debug.ensure_outputs_dir(Path(ibase) / "refiner_self_contained")

        prompt = self._build_self_contained_trace_prompt(trace_steps, trace_answer, executed_tools)
        messages = [{"role": "user", "content": prompt}]
        if out_dir:
            refiner_debug.write_json(
                out_dir,
                "model_input.json",
                {"model": self.planner_model_name, "input": messages},
            )

        raw_output = self._text2text(
            messages, self.planner_model_name, self.planner_api_base, self.planner_api_keys
        )
        if out_dir:
            refiner_debug.write_text(out_dir, "raw_output.txt", raw_output or "")

        parsed = self._extract_json_payload(raw_output)
        parsed_dict = parsed if isinstance(parsed, dict) else None
        if out_dir and parsed_dict is not None:
            refiner_debug.write_json(out_dir, "parsed.json", parsed_dict)
        return raw_output, parsed_dict

    def _build_refiner_prompt(
        self,
        trace_steps: list,
        trace_answer: str,
        diagnosis,
        executed_tools: list,
        planner_output: dict,
    ) -> str:
        diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, indent=2) if isinstance(diagnosis, dict) else str(
            diagnosis
        )
        tool_lines = []
        signal_lines = []
        for item in executed_tools or []:
            if isinstance(item, dict):
                tool_name = item.get("tool", "")
                signal = self._tool_signal_summary_from_output(tool_name, item.get("output", ""))
                if signal:
                    signal_lines.append(f"Step {item.get('step')}: {tool_name} — {signal}")
                safe_output = self._sanitize_tool_output_for_refiner(tool_name, item.get("output", ""))
                tool_lines.append(
                    f"Step {item.get('step')}: {tool_name} — {item.get('purpose', '')}\n{safe_output}"
                )
        tools_block = "\n".join(tool_lines)
        tool_signals_block = "\n".join(signal_lines) if signal_lines else "(no numeric confidence or relevance exposed)"
        refinstr = ""
        if isinstance(planner_output, dict):
            refinstr = str(planner_output.get("refinement_instructions", "") or "")
        return (
            refiner_prompt.strip()
            + "\n\nQUESTION:\n"
            + self._format_question_with_options()
            + "\n\nORIGINAL_TRACE:\n"
            + self._format_trace_steps(trace_steps)
            + "\n\nORIGINAL_ANSWER:\n"
            + trace_answer
            + "\n\nDIAGNOSIS:\n"
            + diagnosis_text
            + "\n\nTOOL_SIGNAL_SUMMARY:\n"
            + tool_signals_block
            + "\n\nTOOL_OUTPUTS:\n"
            + tools_block
            + "\n\nREFINEMENT_INSTRUCTIONS:\n"
            + refinstr
            + "\n\nTRACE_FORMAT: Ordered list of strings; the list position already defines step order.\n"
        )

    def _call_refiner(
        self,
        trace_steps: list,
        trace_answer: str,
        diagnosis,
        executed_tools: list,
        planner_output: dict,
    ):
        ibase = getattr(self, "_refinement_debug_iter_dir", None)
        r_out_dir = None
        if ibase:
            r_out_dir = refiner_debug.ensure_outputs_dir(Path(ibase) / "refiner")

        prompt = self._build_refiner_prompt(
            trace_steps, trace_answer, diagnosis, executed_tools, planner_output or {}
        )
        messages = [{"role": "user", "content": prompt}]
        if r_out_dir:
            refiner_debug.write_json(
                r_out_dir,
                "model_input.json",
                {"model": self.planner_model_name, "input": messages},
            )

        raw_output = self._text2text(
            messages, self.planner_model_name, self.planner_api_base, self.planner_api_keys
        )
        if r_out_dir:
            refiner_debug.write_text(r_out_dir, "raw_output.txt", raw_output or "")

        parsed_output = self._extract_refiner_payload(raw_output)
        repair_raw_output = ""
        if parsed_output is None and str(raw_output or "").strip():
            repair_raw_output = self._retry_malformed_json_response(
                raw_output,
                self.planner_model_name,
                self.planner_api_base,
                self.planner_api_keys,
                schema_name="refiner",
                required_keys=[
                    "refined_trace",
                    "refined_answer",
                    "answer_changed",
                    "changes_made",
                    "unresolved_issues",
                ],
            )
            if r_out_dir:
                refiner_debug.write_text(r_out_dir, "repair_raw_output.txt", repair_raw_output or "")
            parsed_output = self._extract_refiner_payload(repair_raw_output)
        parsed_dict = parsed_output if isinstance(parsed_output, dict) else None
        if parsed_dict is not None:
            current_refined_trace = self._normalize_refined_trace(
                parsed_dict.get("refined_trace"),
                trace_steps,
            )
            if self._trace_needs_self_contained_rewrite(current_refined_trace):
                rewrite_raw, rewrite_dict = self._rewrite_trace_self_contained(
                    current_refined_trace,
                    parsed_dict.get("refined_answer", trace_answer),
                    executed_tools,
                )
                if r_out_dir:
                    refiner_debug.write_text(
                        r_out_dir,
                        "self_contained_rewrite_raw_output.txt",
                        rewrite_raw or "",
                    )
                    if isinstance(rewrite_dict, dict):
                        refiner_debug.write_json(
                            r_out_dir,
                            "self_contained_rewrite_parsed.json",
                            rewrite_dict,
                        )
                if isinstance(rewrite_dict, dict):
                    rewritten_trace = self._normalize_refined_trace(
                        rewrite_dict.get("refined_trace"),
                        current_refined_trace,
                    )
                    rewritten_answer = rewrite_dict.get("refined_answer")
                    if rewritten_trace and not self._trace_needs_self_contained_rewrite(rewritten_trace):
                        parsed_dict["refined_trace"] = rewrite_dict.get("refined_trace")
                        if rewritten_answer is not None and str(rewritten_answer).strip():
                            parsed_dict["refined_answer"] = str(rewritten_answer).strip()
        if r_out_dir and parsed_dict is not None:
            refiner_debug.write_json(r_out_dir, "parsed.json", parsed_dict)

        return raw_output, parsed_dict

    # =====================================================================
    # Trace Generation (cold-start, no initial trace)
    # =====================================================================

    def _build_initial_generation_diagnosis(self) -> dict:
        return {
            "verdict": "FAIL",
            "answer_correct": False,
            "trace_quality_scores": {
                "perceptual_correctness": 0.0,
                "temporal_accuracy": 0.0,
                "logical_coherence": 0.0,
                "completeness": 0.0,
            },
            "error_categories": [
                {
                    "type": "INCOMPLETE_TRACE",
                    "step_index": None,
                    "description": (
                        "No initial reasoning trace or answer is available. Decompose the question "
                        "into answer-critical subgoals, gather the minimal tool evidence needed to "
                        "answer it from scratch, and prepare refinement instructions that let the "
                        "refiner synthesize a complete first trace."
                    ),
                    "severity": "HIGH",
                    "suggested_tools": [],
                    "evidence": None,
                }
            ],
            "confidence": 1.0,
            "summary": (
                "Cold-start generation mode: ORIGINAL_TRACE and ORIGINAL_ANSWER are intentionally "
                "empty. Plan tool calls from the question alone, then synthesize the first "
                "tool-grounded trace."
            ),
            "generation_mode": "cold_start",
        }

    def _planner_output_for_initial_generation(self, planner_output: dict | None) -> dict:
        out = dict(planner_output or {})
        existing = str(out.get("refinement_instructions", "") or "").strip()
        generation_note = (
            "GENERATION MODE: ORIGINAL_TRACE and ORIGINAL_ANSWER are intentionally empty. "
            "Do not patch nonexistent steps. Instead synthesize a complete initial trace from "
            "the TOOL_OUTPUTS, decomposed into clear question-aligned reasoning steps. Every "
            "media-grounded claim must preserve tool provenance inline in the trace itself. "
            "Write the final trace as a compact reader-facing argument rather than a tool log: "
            "prefer 3-6 short steps in claim-first order, group multi-part questions by subgoal, "
            "and include numeric confidence or relevance inline whenever the tool outputs expose it. "
            "If the gathered evidence remains partial or ambiguous, state that limitation "
            "explicitly in the trace and unresolved_issues rather than forcing unsupported details."
        )
        out["refinement_instructions"] = (
            generation_note if not existing else generation_note + "\n\n" + existing
        )
        return out

    def _build_generator_round_prompt(
        self,
        round_idx: int,
        max_rounds: int,
        tool_outputs_so_far: list,
    ) -> str:
        question_block = self._format_question_with_options()
        # Intentionally disabled for artifact-free runs: do not inject
        # PREPROCESSED_ARTIFACTS into generator prompts.

        tool_history = ""
        if tool_outputs_so_far:
            lines = []
            for i, item in enumerate(tool_outputs_so_far, 1):
                tool_name = item.get("tool", "")
                signal = self._tool_signal_summary_from_output(tool_name, item.get("output", ""))
                signal_line = f"  Signal summary: {signal}\n" if signal else ""
                lines.append(
                    f"Round {i}: {tool_name}({json.dumps(item.get('arguments', {}), ensure_ascii=False)})\n"
                    f"  Purpose: {item.get('purpose', '')}\n"
                    f"{signal_line}"
                    f"  Output: {item.get('output', '')}"
                )
            tool_history = "\n\n".join(lines)

        force_trace = ""
        if round_idx >= max_rounds - 1:
            force_trace = (
                "\n\nFINAL ROUND — you MUST output type \"trace\" now. Synthesise the "
                "best possible trace and answer from the evidence collected so far."
            )

        cached_summary = str(getattr(self, "_video_caption_summary", "") or "").strip()

        return (
            trace_generator_prompt.strip()
            + "\n\nQUESTION:\n"
            + question_block
            + "\n\nVIDEO_PATH:\n"
            + str(self.video_path)
            + "\n\nVIDEO_DURATION:\n"
            + str(self.duration) + " seconds"
            + (
                "\n\nVIDEO_CAPTION_SUMMARY:\n" + cached_summary
                if cached_summary
                else ""
            )
            + "\n\nROUND:\n"
            + f"{round_idx + 1}/{max_rounds}"
            + "\n\nPREVIOUS_TOOL_OUTPUTS:\n"
            + (tool_history or "(none yet — this is the first round)")
            + force_trace
        )

    def _extract_generator_output(self, raw_text: str) -> dict | None:
        """Parse generator response into either a tool_call or trace dict."""
        parsed = self._extract_generator_payload(raw_text)
        if not isinstance(parsed, dict):
            return None
        output_type = str(parsed.get("type", "")).strip().lower()
        if output_type == "trace":
            return {
                "type": "trace",
                "trace_steps": [str(s).strip() for s in (parsed.get("trace_steps") or []) if str(s).strip()],
                "answer": str(parsed.get("answer", "")).strip(),
            }
        if output_type == "tool_call":
            return {
                "type": "tool_call",
                "tool": str(parsed.get("tool", "")).strip(),
                "arguments": parsed.get("arguments", {}) if isinstance(parsed.get("arguments"), dict) else {},
                "purpose": str(parsed.get("purpose", "")).strip(),
            }
        return None

    def _execute_single_tool_call(self, tool_call: dict) -> str:
        """Execute one tool call using the existing refiner tool infrastructure."""
        tool_name = str(tool_call.get("tool", "")).strip()
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        arguments = self._replace_runtime_placeholders(arguments)

        # Inject video_path if the tool expects it and it's missing or unresolved.
        if "video_path" in self._get_tool_argument_names(tool_name) and not str(arguments.get("video_path", "") or "").strip():
            arguments["video_path"] = self.video_path

        unresolved_refs = self._collect_unresolved_step_refs(arguments)
        if unresolved_refs:
            blocked_steps = []
            for ref in unresolved_refs:
                match = self._STEP_REF_RE.fullmatch(str(ref).strip())
                if not match:
                    continue
                try:
                    blocked_steps.append(int(match.group(1)))
                except (TypeError, ValueError):
                    continue
            result = self._dependency_blocked_result(
                tool_name,
                "unresolved dependency outputs remain in arguments",
                blocked_steps=sorted(set(blocked_steps)) or None,
                unresolved_refs=unresolved_refs,
            )
            return self._format_refine_tool_result(tool_name, arguments, result)

        try:
            arguments = self._validate_tool_arguments(tool_name, arguments)
        except Exception as e:
            result = self._tool_validation_error_result(tool_name, e)
            return self._format_refine_tool_result(tool_name, arguments, result)

        try:
            return self._execute_refine_tool_call(tool_name, arguments)
        except Exception as exc:
            return f"Error executing {tool_name}: {exc}"

    def _get_tool_argument_names(self, tool_name: str) -> set:
        """Return the set of expected argument names for a tool."""
        known = {
            "temporal_grounder": {"video_path", "query"},
            "frame_retriever": {"video_path", "query", "timestamps", "time_range", "num_frames"},
            "asr": {"video_path", "start_time", "end_time"},
            "audio_grounder": {"video_path", "query", "start_time", "end_time"},
            "ocr": {"frame_path", "timestamp"},
            "spatial_grounder": {"frame_path", "timestamp", "query"},
            "counter": {"frame_path", "timestamp", "query", "exemplar_paths"},
            "dense_captioner": {"video_path", "start_time", "end_time", "granularity"},
            "action_recognizer": {"video_path", "start_time", "end_time"},
            "chart_analyzer": {"frame_path", "timestamp", "query"},
        }
        return known.get(tool_name, set())

    def _generator_max_rounds(self) -> int:
        raw = str(os.environ.get("TRACE_GENERATOR_MAX_ROUNDS", "") or "").strip()
        try:
            value = int(raw) if raw else 4
        except Exception:
            value = 4
        return max(1, min(value, 8))

    def _call_trace_generator_one_shot_fallback(self, generation_diagnosis: dict, debug_base: Path | None):
        generation_record = {
            "phase": "initial_trace_generation_fallback",
            "diagnosis": generation_diagnosis,
        }

        print("[Generation Planner Fallback] Generating initial evidence plan...")
        planner_raw, planner_output = self._call_planner(
            [],
            "",
            generation_diagnosis,
            iteration=0,
            history=[],
            max_iterations=1,
        )
        print(f"\n[Generation Planner Output]\n{planner_raw}\n")

        print("[Generation Executor Fallback] Running planned tool calls...")
        executed_tools = self._execute_refine_plan(planner_output if planner_output is not None else {})
        for item in executed_tools:
            print(f"  Step {item['step']} - {item['tool']}")

        generation_planner_output = self._planner_output_for_initial_generation(planner_output)

        print("[Generation Refiner Fallback] Synthesizing initial trace...")
        refiner_raw, refiner_output = self._call_refiner(
            [],
            "",
            generation_diagnosis,
            executed_tools,
            generation_planner_output,
        )
        print(f"\n[Generation Refiner Output]\n{refiner_raw}\n")

        generated_steps = []
        generated_answer = ""
        if isinstance(refiner_output, dict):
            generated_steps = self._normalize_refined_trace(
                refiner_output.get("refined_trace"),
                [],
            )
            raw_answer = refiner_output.get("refined_answer", "")
            if raw_answer is not None and str(raw_answer).strip():
                generated_answer = str(raw_answer).strip()
            elif generated_steps:
                generated_answer = (self._extract_trace_answer(generated_steps) or "").strip()

        if not generated_steps:
            generated_answer = ""
            generated_steps = []
            for item in executed_tools:
                generated_steps.append(
                    f"{item.get('tool', 'tool')} was called for {item.get('purpose', '')}. "
                    f"Reported output: {str(item.get('output', '') or '')[:500]}"
                )
            generated_steps.append(
                "Unable to synthesize a complete initial trace from the gathered evidence."
            )

        generation_record.update(
            {
                "planner_raw": planner_raw,
                "planner_output": planner_output,
                "executed_tools": executed_tools,
                "refiner_raw": refiner_raw,
                "refiner_output": refiner_output,
            }
        )
        if debug_base:
            refiner_debug.write_json(debug_base, "generation_fallback_summary.json", generation_record)
        return generated_steps, generated_answer, generation_record

    def _call_trace_generator(self):
        """Iterative trace generation from an empty initial trace, with planner fallback."""
        print("\n" + "=" * 70)
        print("Starting Trace Generation (cold-start, no initial trace)")
        print("=" * 70 + "\n")

        generation_diagnosis = self._build_initial_generation_diagnosis()
        generation_rounds = []
        tool_outputs_so_far = []

        prev_iter_dir = getattr(self, "_refinement_debug_iter_dir", None)
        debug_base = None
        if self.refinement_debug_root:
            session_base = self._ensure_refinement_debug_session_base()
            if session_base:
                debug_base = Path(session_base) / "generation" / "initial_trace"
                debug_base.mkdir(parents=True, exist_ok=True)
                self._refinement_debug_iter_dir = str(debug_base)

        try:
            max_rounds = self._generator_max_rounds()
            messages = None

            for round_idx in range(max_rounds):
                prompt = self._build_generator_round_prompt(round_idx, max_rounds, tool_outputs_so_far)
                messages = [{"role": "user", "content": prompt}]
                print(f"[Generation Round {round_idx + 1}/{max_rounds}] Requesting next action...")
                generator_raw = self._text2text(
                    messages,
                    self.planner_model_name,
                    self.planner_api_base,
                    self.planner_api_keys,
                )
                generator_output = self._extract_generator_output(generator_raw)

                repaired_raw = ""
                if generator_output is None and str(generator_raw or "").strip():
                    repaired_raw = self._retry_malformed_json_response(
                        generator_raw,
                        self.planner_model_name,
                        self.planner_api_base,
                        self.planner_api_keys,
                        "trace generator response",
                        required_keys=["type"],
                    )
                    generator_output = self._extract_generator_output(repaired_raw)

                round_record = {
                    "round": round_idx + 1,
                    "prompt": prompt,
                    "generator_raw": generator_raw,
                    "generator_repair_raw": repaired_raw,
                    "generator_output": generator_output,
                }

                if generator_output is None:
                    generation_rounds.append(round_record)
                    print("[Generation] Could not parse generator JSON; falling back to planner/refiner path.")
                    break

                if generator_output.get("type") == "tool_call":
                    executed_output = self._execute_single_tool_call(generator_output)
                    executed_record = {
                        "tool": generator_output.get("tool", ""),
                        "arguments": dict(generator_output.get("arguments") or {}),
                        "purpose": generator_output.get("purpose", ""),
                        "output": executed_output,
                    }
                    tool_outputs_so_far.append(executed_record)
                    round_record["executed_tool"] = executed_record
                    generation_rounds.append(round_record)
                    if debug_base:
                        refiner_debug.write_json(
                            debug_base,
                            f"round_{round_idx + 1}.json",
                            round_record,
                        )
                    continue

                trace_steps = self._normalize_refined_trace(generator_output.get("trace_steps"), [])
                trace_answer = str(generator_output.get("answer", "") or "").strip()
                if not trace_answer and trace_steps:
                    trace_answer = (self._extract_trace_answer(trace_steps) or "").strip()
                round_record["final_trace"] = {"steps": trace_steps, "answer": trace_answer}
                generation_rounds.append(round_record)
                if debug_base:
                    refiner_debug.write_json(
                        debug_base,
                        f"round_{round_idx + 1}.json",
                        round_record,
                    )

                if trace_steps:
                    if debug_base:
                        refiner_debug.write_json(
                            debug_base,
                            "generation_summary.json",
                            {"mode": "iterative", "rounds": generation_rounds},
                        )
                    return trace_steps, trace_answer, generation_rounds

                print("[Generation] Trace output was empty; falling back to planner/refiner path.")
                break

            generated_steps, generated_answer, fallback_record = self._call_trace_generator_one_shot_fallback(
                generation_diagnosis,
                debug_base,
            )
            generation_rounds.append(fallback_record)
            if debug_base:
                refiner_debug.write_json(
                    debug_base,
                    "generation_summary.json",
                    {"mode": "iterative_with_fallback", "rounds": generation_rounds},
                )
            return generated_steps, generated_answer, generation_rounds
        finally:
            self._refinement_debug_iter_dir = prev_iter_dir
