import json
import os
import re
from pathlib import Path

import refiner_debug
from refine_prompt import planner_prompt, refiner_prompt, trace_generator_prompt, verifier_prompt


class RefinerAgentsMixin:
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
        parsed = self._extract_json_payload(tail)
        return parsed if isinstance(parsed, dict) else None

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
        artifacts_text = json.dumps(self._get_preprocessed_artifacts(), ensure_ascii=False, indent=2)
        ctx = self._iteration_context_block(iteration, max_iterations, history or [])
        last_tools = ""
        if history:
            last_tools = json.dumps(history[-1].get("tools_executed", []), ensure_ascii=False, indent=2)
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
            + "\n\nPREPROCESSED_ARTIFACTS:\n"
            + artifacts_text
            + "\n\nPREVIOUS_TOOL_RESULTS_SUMMARY (last iteration tools + confidence):\n"
            + (last_tools or "[]")
        )

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
        if p_out_dir and parsed_dict is not None:
            refiner_debug.write_json(p_out_dir, "plan.json", parsed_dict)
        if parsed_dict is None:
            raise RuntimeError("Planner returned malformed JSON and could not be repaired.")

        return raw_output, parsed_dict

    def _normalize_refined_trace(self, raw, fallback: list) -> list:
        if isinstance(raw, list):
            out = []
            for item in raw:
                text = str(item).strip()
                m = re.match(r"^\d+\.\s*(.*)$", text)
                out.append(m.group(1) if m else text)
            return out
        if isinstance(raw, str):
            lines = [ln.rstrip() for ln in raw.split("\n")]
            out = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^\d+\.\s*(.*)$", line)
                out.append(m.group(1) if m else line)
            return out if out else [raw]
        return list(fallback)

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
        for item in executed_tools or []:
            if isinstance(item, dict):
                tool_lines.append(
                    f"Step {item.get('step')}: {item.get('tool')} — {item.get('purpose', '')}\n{item.get('output', '')}"
                )
        tools_block = "\n".join(tool_lines)
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
            + "\n\nTOOL_OUTPUTS:\n"
            + tools_block
            + "\n\nREFINEMENT_INSTRUCTIONS:\n"
            + refinstr
            + "\n\nTRACE_FORMAT: Numbered list of strings (same style as ORIGINAL_TRACE).\n"
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
        parsed_dict = parsed_output if isinstance(parsed_output, dict) else None
        if r_out_dir and parsed_dict is not None:
            refiner_debug.write_json(r_out_dir, "parsed.json", parsed_dict)

        return raw_output, parsed_dict

    # =====================================================================
    # Trace Generation (cold-start, no initial trace)
    # =====================================================================

    def _build_generator_round_prompt(
        self,
        round_idx: int,
        max_rounds: int,
        tool_outputs_so_far: list,
    ) -> str:
        question_block = self._format_question_with_options()
        artifacts_text = json.dumps(
            self._get_preprocessed_artifacts(), ensure_ascii=False, indent=2
        )

        tool_history = ""
        if tool_outputs_so_far:
            lines = []
            for i, item in enumerate(tool_outputs_so_far, 1):
                lines.append(
                    f"Round {i}: {item['tool']}({json.dumps(item.get('arguments', {}), ensure_ascii=False)})\n"
                    f"  Purpose: {item.get('purpose', '')}\n"
                    f"  Output: {item.get('output', '')}"
                )
            tool_history = "\n\n".join(lines)

        force_trace = ""
        if round_idx >= max_rounds - 1:
            force_trace = (
                "\n\nFINAL ROUND — you MUST output type \"trace\" now. Synthesise the "
                "best possible trace and answer from the evidence collected so far."
            )

        return (
            trace_generator_prompt.strip()
            + "\n\nQUESTION:\n"
            + question_block
            + "\n\nVIDEO_PATH:\n"
            + str(self.video_path)
            + "\n\nVIDEO_DURATION:\n"
            + str(self.duration) + " seconds"
            + "\n\nPREPROCESSED_ARTIFACTS:\n"
            + artifacts_text
            + "\n\nROUND:\n"
            + f"{round_idx + 1}/{max_rounds}"
            + "\n\nPREVIOUS_TOOL_OUTPUTS:\n"
            + (tool_history or "(none yet — this is the first round)")
            + force_trace
        )

    def _extract_generator_output(self, raw_text: str) -> dict | None:
        """Parse generator response into either a tool_call or trace dict."""
        parsed = self._extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            return None

        output_type = str(parsed.get("type", "")).strip().lower()

        if output_type == "trace":
            steps = parsed.get("trace_steps", [])
            if isinstance(steps, list) and steps:
                return {
                    "type": "trace",
                    "trace_steps": [str(s).strip() for s in steps if str(s).strip()],
                    "answer": str(parsed.get("answer", "")).strip(),
                }

        if output_type == "tool_call":
            tool = str(parsed.get("tool", "")).strip()
            if tool:
                return {
                    "type": "tool_call",
                    "tool": tool,
                    "arguments": parsed.get("arguments", {}) if isinstance(parsed.get("arguments"), dict) else {},
                    "purpose": str(parsed.get("purpose", "")).strip(),
                }

        # Fallback: if it has trace_steps, treat as trace even without explicit type
        if isinstance(parsed.get("trace_steps"), list) and parsed["trace_steps"]:
            return {
                "type": "trace",
                "trace_steps": [str(s).strip() for s in parsed["trace_steps"] if str(s).strip()],
                "answer": str(parsed.get("answer", "")).strip(),
            }
        # Fallback: if it has tool field, treat as tool_call
        if parsed.get("tool"):
            return {
                "type": "tool_call",
                "tool": str(parsed["tool"]).strip(),
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

        # Inject video_path if the tool expects it and it's not provided
        if "video_path" in self._get_tool_argument_names(tool_name) and "video_path" not in arguments:
            arguments["video_path"] = self.video_path

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
            "frame_retriever": {"video_path", "query", "timestamps", "num_frames"},
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

    def _call_trace_generator(self, max_rounds: int = 10):
        """Iterative trace generation: call tools one at a time, produce trace when ready."""
        print("\n" + "=" * 70)
        print("Starting Trace Generation (cold-start, no initial trace)")
        print("=" * 70 + "\n")

        debug_base = None
        if self.refinement_debug_root:
            stem = refiner_debug.sanitize_path_component(Path(self.video_path).stem)
            debug_base = Path(self.refinement_debug_root) / stem / "generation"
            debug_base.mkdir(parents=True, exist_ok=True)

        tool_outputs = []
        all_rounds = []

        for round_idx in range(max_rounds):
            print(f"\n[Generation Round {round_idx + 1}/{max_rounds}]")

            prompt = self._build_generator_round_prompt(
                round_idx, max_rounds, tool_outputs
            )
            messages = [{"role": "user", "content": prompt}]

            if debug_base:
                round_dir = debug_base / f"round_{round_idx + 1:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                refiner_debug.write_json(
                    str(round_dir), "model_input.json",
                    {"model": self.planner_model_name, "input": messages},
                )

            raw_output = self._text2text(
                messages,
                self.planner_model_name,
                self.planner_api_base,
                self.planner_api_keys,
            )
            print(f"[Generator Output]\n{raw_output}\n")

            if debug_base:
                refiner_debug.write_text(
                    str(round_dir), "raw_output.txt", raw_output or ""
                )

            parsed = self._extract_generator_output(raw_output)

            if parsed is None:
                print(f"  [Warning] Could not parse generator output, retrying...")
                repair_raw = self._retry_malformed_json_response(
                    raw_output,
                    self.planner_model_name,
                    self.planner_api_base,
                    self.planner_api_keys,
                    schema_name="trace_generator",
                    required_keys=["type"],
                )
                parsed = self._extract_generator_output(repair_raw)

            if parsed is None:
                print(f"  [Error] Unparseable output on round {round_idx + 1}, skipping.")
                all_rounds.append({
                    "round": round_idx + 1,
                    "raw_output": raw_output,
                    "parsed": None,
                    "type": "error",
                })
                continue

            if parsed["type"] == "trace":
                print(f"  [Trace produced on round {round_idx + 1}]")
                print(f"  Steps: {len(parsed['trace_steps'])}")
                print(f"  Answer: {parsed['answer']}")
                all_rounds.append({
                    "round": round_idx + 1,
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "type": "trace",
                })
                if debug_base:
                    refiner_debug.write_json(
                        str(round_dir), "final_trace.json", parsed
                    )
                return parsed["trace_steps"], parsed["answer"], all_rounds

            # type == "tool_call"
            tool_name = parsed["tool"]
            print(f"  [Tool call] {tool_name}: {parsed.get('purpose', '')}")

            tool_output = self._execute_single_tool_call(parsed)
            print(f"  [Tool output] {tool_output[:200]}..." if len(tool_output) > 200 else f"  [Tool output] {tool_output}")

            tool_record = {
                "tool": tool_name,
                "arguments": parsed.get("arguments", {}),
                "purpose": parsed.get("purpose", ""),
                "output": tool_output,
            }
            tool_outputs.append(tool_record)

            all_rounds.append({
                "round": round_idx + 1,
                "raw_output": raw_output,
                "parsed": parsed,
                "type": "tool_call",
                "tool_output": tool_output,
            })
            if debug_base:
                refiner_debug.write_text(
                    str(round_dir), "tool_output.txt", tool_output
                )

        # If we exhausted all rounds without a trace, force one final attempt
        print("\n[Warning] Max generation rounds reached without trace output.")
        print("[Forcing final trace synthesis...]")
        prompt = self._build_generator_round_prompt(
            max_rounds - 1, max_rounds, tool_outputs
        )
        messages = [{"role": "user", "content": prompt}]
        raw_output = self._text2text(
            messages,
            self.planner_model_name,
            self.planner_api_base,
            self.planner_api_keys,
        )
        parsed = self._extract_generator_output(raw_output)
        if parsed and parsed["type"] == "trace":
            all_rounds.append({
                "round": max_rounds + 1,
                "raw_output": raw_output,
                "parsed": parsed,
                "type": "trace",
            })
            return parsed["trace_steps"], parsed["answer"], all_rounds

        # Last resort: synthesize a minimal trace from tool outputs
        fallback_steps = []
        for i, item in enumerate(tool_outputs, 1):
            fallback_steps.append(
                f"Tool {item['tool']} was called: {item['purpose']}. "
                f"Result: {item['output'][:500]}"
            )
        fallback_steps.append("Unable to derive a confident answer from the evidence gathered.")
        all_rounds.append({
            "round": max_rounds + 1,
            "type": "fallback",
        })
        return fallback_steps, "", all_rounds
