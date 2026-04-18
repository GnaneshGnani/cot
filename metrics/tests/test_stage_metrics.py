import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "metrics"
EVAL_DIR = ROOT / "VideoDeepResearch" / "eval"
for path in (METRICS_DIR, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answer_sufficiency import compute_stage_metric as compute_answer_sufficiency
from execution_consistency import compute_stage_metric as compute_execution_consistency
from internal_coherence import compute_stage_metric as compute_internal_coherence
from stage_loader import load_stage_samples
from stage_metrics_common import build_stage_record
from tool_validity import compute_stage_metric as compute_tool_validity
import verifier_metrics


class DummyJudge(object):
    def __init__(self, response):
        self.response = response

    def _vlm_summarize_text(self, _prompt):
        return self.response


class DummyEmbeddingModel(object):
    def encode(self, steps):
        return np.eye(len(steps))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


class StageMetricTests(unittest.TestCase):
    def test_vmqa_stage_local_correctness_matches_letter_and_text(self):
        sample_dir = (
            ROOT
            / "VideoDeepResearch"
            / "eval"
            / "results_generated_full_context_videomathqa_10pct"
            / "03bb5639-4b1e-48ae-8542-905647627ffe"
        )
        sample = load_stage_samples(sample_dir)[0]
        stage_record = build_stage_record(sample, "ref2")
        self.assertEqual(stage_record["gold_answer"], "E")
        self.assertEqual(stage_record["proposed_answer"], "E. 10")
        self.assertTrue(stage_record["is_correct"])

    def test_stage_resolution_and_terminal_verifier_sourcing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            generated_dir = root / "generated_terminal"
            _write_json(
                generated_dir / "meta.json",
                {
                    "video_path": "videos/a.mp4",
                    "question": "Q1",
                    "options": ["A. 1", "B. 2"],
                    "answer": "B",
                    "initial_trace_steps": ["init"],
                    "final_trace": {"steps": ["generated final"]},
                    "final_answer": "B. 2",
                    "terminal_stage": "generated",
                    "final_verifier_output": {"verdict": "PASS", "trace_quality_scores": {"logical_coherence": 8, "completeness": 8, "factual_correctness": 8, "reasoning_order": 8}},
                },
            )
            _write_json(generated_dir / "generated_trace.json", {"trace_steps": ["generated final"], "answer": "B. 2"})

            ref1_dir = root / "ref1_terminal"
            _write_json(
                ref1_dir / "meta.json",
                {
                    "video_path": "videos/b.mp4",
                    "question": "Q2",
                    "options": ["A. 1", "B. 2"],
                    "answer": "B",
                    "initial_trace_steps": ["init"],
                    "final_trace": {"steps": ["ref1 final"]},
                    "final_answer": "B. 2",
                    "terminal_stage": "ref1",
                    "final_verifier_output": {"verdict": "PASS", "trace_quality_scores": {"logical_coherence": 9, "completeness": 9, "factual_correctness": 9, "reasoning_order": 9}},
                },
            )
            _write_json(ref1_dir / "generated_trace.json", {"trace_steps": ["generated"], "answer": "A"})
            _write_json(
                ref1_dir / "refinement_1.json",
                {
                    "verifier_output": {"verdict": "FAIL", "trace_quality_scores": {"logical_coherence": 5, "completeness": 5, "factual_correctness": 5, "reasoning_order": 5}},
                    "refiner_output": {"refined_trace": ["ref1 final"], "refined_answer": "B. 2"},
                    "planner_output": {"tool_calls": []},
                    "executed_tools": [],
                },
            )

            ref2_dir = root / "ref2_terminal"
            _write_json(
                ref2_dir / "meta.json",
                {
                    "video_path": "videos/c.mp4",
                    "question": "Q3",
                    "options": ["A. 1", "B. 2"],
                    "answer": "B",
                    "initial_trace_steps": ["init"],
                    "final_trace": {"steps": ["ref2 final"]},
                    "final_answer": "B. 2",
                },
            )
            _write_json(ref2_dir / "generated_trace.json", {"trace_steps": ["generated"], "answer": "A"})
            _write_json(
                ref2_dir / "refinement_1.json",
                {
                    "verifier_output": {"verdict": "FAIL", "trace_quality_scores": {"logical_coherence": 4, "completeness": 4, "factual_correctness": 4, "reasoning_order": 4}},
                    "refiner_output": {"refined_trace": ["ref1"], "refined_answer": "A"},
                },
            )
            _write_json(
                ref2_dir / "refinement_2.json",
                {
                    "verifier_output": {"verdict": "FAIL", "trace_quality_scores": {"logical_coherence": 6, "completeness": 6, "factual_correctness": 6, "reasoning_order": 6}},
                    "refiner_output": {"refined_trace": ["ref2 final"], "refined_answer": "B. 2"},
                },
            )

            samples = {
                sample["source_dir"]: sample
                for sample in load_stage_samples(root)
            }

            generated_sample = samples[str(generated_dir.resolve())]
            self.assertEqual(generated_sample["terminal_stage"], "generated")
            self.assertTrue(generated_sample["stages"]["generated"]["is_terminal"])
            self.assertIsNotNone(generated_sample["stages"]["generated"]["stored_verifier_output"])

            ref1_sample = samples[str(ref1_dir.resolve())]
            self.assertEqual(ref1_sample["terminal_stage"], "ref1")
            self.assertIsNotNone(ref1_sample["stages"]["generated"]["stored_verifier_output"])
            self.assertIsNotNone(ref1_sample["stages"]["ref1"]["stored_verifier_output"])

            ref2_sample = samples[str(ref2_dir.resolve())]
            self.assertEqual(ref2_sample["terminal_stage"], "ref2")
            self.assertIsNotNone(ref2_sample["stages"]["generated"]["stored_verifier_output"])
            self.assertIsNotNone(ref2_sample["stages"]["ref1"]["stored_verifier_output"])
            self.assertIsNone(ref2_sample["stages"]["ref2"]["stored_verifier_output"])

    def test_metric_applicability_for_initial_trace(self):
        sample_dir = (
            ROOT
            / "VideoDeepResearch"
            / "eval"
            / "results_generated_full_context_videomathqa_10pct"
            / "03bb5639-4b1e-48ae-8542-905647627ffe"
        )
        sample = load_stage_samples(sample_dir)[0]
        stage_record = build_stage_record(sample, "initial")

        sufficiency = compute_answer_sufficiency(stage_record, DummyJudge("Yes"))
        coherence = compute_internal_coherence(stage_record, DummyEmbeddingModel())
        execution = compute_execution_consistency(stage_record)
        tool = compute_tool_validity(stage_record)

        self.assertTrue(sufficiency["applicable"])
        self.assertTrue(coherence["applicable"])
        self.assertFalse(execution["applicable"])
        self.assertEqual(execution["skipped_reason"], "missing_planner_and_execution_artifacts")
        self.assertFalse(tool["applicable"])
        self.assertEqual(tool["skipped_reason"], "missing_planner_and_execution_artifacts")

    def test_generic_jsonl_loader_keeps_same_video_different_questions_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.jsonl"
            rows = [
                {"video_path": "videos/shared.mp4", "question": "Question A", "answer": "A", "trace": ["step one"]},
                {"video_path": "videos/shared.mp4", "question": "Question B", "answer": "B", "trace": ["step two"]},
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            samples = load_stage_samples(path)
            self.assertEqual(len(samples), 2)
            self.assertNotEqual(samples[0]["sample_id"], samples[1]["sample_id"])

    def test_answer_source_overrides_work(self):
        sample = {
            "sample_id": "sample-1",
            "source_dir": "/tmp/sample-1",
            "video_path": "videos/sample.mp4",
            "question": "What is correct?",
            "question_id": "qid-1",
            "options": ["A. wrong", "B. right"],
            "gold_answer": "B",
            "final_answer": "B. right",
            "terminal_stage": "ref2",
            "metadata": {},
            "stages": {
                "generated": {"available": True, "trace_steps": ["g"], "stage_local_answer": "A", "executed_tools": []},
                "ref1": {"available": True, "trace_steps": ["r1"], "stage_local_answer": "A", "executed_tools": []},
                "ref2": {"available": True, "trace_steps": ["r2"], "stage_local_answer": "B. right", "executed_tools": [], "is_terminal": True},
            },
        }

        generated_default = build_stage_record(sample, "generated")
        generated_terminal = build_stage_record(sample, "generated", {"generated": "terminal_final"})
        generated_benchmark = build_stage_record(sample, "generated", {"generated": "benchmark"})

        self.assertEqual(generated_default["proposed_answer"], "A")
        self.assertEqual(generated_terminal["proposed_answer"], "B. right")
        self.assertEqual(generated_benchmark["proposed_answer"], "B")

    def test_hybrid_verifier_mode_works_for_old_runs_via_fallback(self):
        vmqa_sample = load_stage_samples(
            ROOT / "VideoDeepResearch" / "eval" / "results_generated_full_context_videomathqa_10pct",
            max_samples=1,
        )[0]
        ovb_sample = load_stage_samples(
            ROOT / "VideoDeepResearch" / "eval" / "results_omnivideobench",
            max_samples=1,
        )[0]

        for sample in (vmqa_sample, ovb_sample):
            stage_name = sample["terminal_stage"]
            stage_record = build_stage_record(sample, stage_name)
            with mock.patch.object(
                verifier_metrics,
                "resolve_verifier",
                return_value={
                    "verifier_mode_used": "offline",
                    "verifier_raw": "{\"verdict\": \"PASS\"}",
                    "verifier_output": {
                        "verdict": "PASS",
                        "confidence": 0.9,
                        "answer_correct": True,
                        "error_categories": [],
                        "trace_quality_scores": {
                            "logical_coherence": 8,
                            "completeness": 8,
                            "factual_correctness": 8,
                            "reasoning_order": 8,
                        },
                    },
                    "model": "mock-model",
                },
            ):
                metric = verifier_metrics.compute_stage_metric(stage_record, verifier_mode="hybrid")
            self.assertTrue(metric["applicable"])
            self.assertEqual(metric["verifier_mode_used"], "offline")

if __name__ == "__main__":
    unittest.main()
