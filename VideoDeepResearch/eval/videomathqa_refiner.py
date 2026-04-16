import argparse
import json
import os
import random
import sys
from pathlib import Path

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from refiner import VideoQADemo


DEFAULT_VIDEOMATHQA_ROOT = Path("/nfs-stor/ghazi.ahmad/VideoMathQA")
DEFAULT_ANNOTATION_FILE = DEFAULT_VIDEOMATHQA_ROOT / "mcq.json"
DEFAULT_RESULTS_DIR = Path(_eval_dir) / "results_generated_full_context_videomathqa_10pct"
DEFAULT_DEBUG_DIR = Path(_eval_dir) / "debug_videomathqa_10pct"
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_SAMPLE_SEED = 1337


def _env_list(name: str, fallback: str = ""):
    raw = (os.environ.get(name, fallback) or "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_steps_field(value):
    if value is None:
        return None

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]

    if isinstance(parsed, list):
        steps = [str(item).strip() for item in parsed if str(item).strip()]
        return steps or None

    if isinstance(parsed, dict):
        def _sort_key(item):
            key = str(item[0])
            if key.isdigit():
                return (0, int(key))
            return (1, key)

        steps = []
        for _, step_text in sorted(parsed.items(), key=_sort_key):
            step_str = str(step_text).strip()
            if step_str:
                steps.append(step_str)
        return steps or None

    step_str = str(parsed).strip()
    return [step_str] if step_str else None


def _load_videomathqa_samples(annotation_path: Path, dataset_root: Path):
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {annotation_path}")

    videos_dir = dataset_root / "videos"
    samples = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Expected an object at index {index} in {annotation_path}")

        video_id = str(item.get("videoID", "")).strip()
        if not video_id:
            raise ValueError(f"Missing videoID at index {index} in {annotation_path}")

        sample = dict(item)
        sample["source_index"] = index
        sample["video_path"] = str((videos_dir / f"{video_id}.mp4").resolve())
        sample["initial_trace_steps"] = _parse_steps_field(item.get("steps"))
        samples.append(sample)

    return samples


def _select_random_subset(samples, sample_ratio: float, sample_seed: int):
    if not samples:
        return []

    sample_count = max(1, int(len(samples) * sample_ratio))
    sample_count = min(sample_count, len(samples))

    rng = random.Random(sample_seed)
    chosen_indices = sorted(rng.sample(range(len(samples)), sample_count))
    return [samples[index] for index in chosen_indices]


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _clear_stale_split_outputs(out_dir: Path):
    if not out_dir.is_dir():
        return

    for path in out_dir.glob("refinement_*.json"):
        try:
            path.unlink()
        except OSError:
            pass

    generated_trace = out_dir / "generated_trace.json"
    if generated_trace.is_file():
        try:
            generated_trace.unlink()
        except OSError:
            pass


def _save_result_dir(record, video_path, out_dir: Path, sample_ratio: float, sample_seed: int):
    question = str(record.get("question", "") or "").strip()
    options = list(record.get("options") or [])
    input_answer = record.get("answer")
    initial_trace_steps = record.get("initial_trace_steps")

    meta = {
        "videoID": str(record.get("videoID", "") or "").strip(),
        "question_id": record.get("question_id"),
        "category": record.get("category"),
        "length": record.get("length"),
        "source_index": record.get("source_index"),
        "sample_ratio": sample_ratio,
        "sample_seed": sample_seed,
        "video_path": str(record.get("video_path", "") or video_path or "").strip(),
        "question": question,
        "options": options,
        "answer": input_answer,
        "initial_trace_steps": initial_trace_steps,
    }
    if record.get("refiner_error"):
        meta["refiner_error"] = record["refiner_error"]

    rr = record.get("refiner_result")
    if isinstance(rr, dict):
        meta.update(
            {
                "trace_generated": rr.get("trace_generated"),
                "initial_answer": rr.get("initial_answer"),
                "final_trace": rr.get("final_trace"),
                "final_answer": rr.get("final_answer"),
                "is_correct": rr.get("is_correct"),
                "max_iterations": rr.get("max_iterations"),
                "refinement_debug_root": rr.get("refinement_debug_root"),
            }
        )
        if rr.get("trace_generated") and rr.get("generated_trace") is not None:
            _write_json(out_dir / "generated_trace.json", rr["generated_trace"])

    _write_json(out_dir / "meta.json", meta)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "annotation_file",
        nargs="?",
        default=str(DEFAULT_ANNOTATION_FILE),
        help="Path to VideoMathQA mcq.json",
    )
    parser.add_argument(
        "--videomathqa-root",
        type=str,
        default=str(DEFAULT_VIDEOMATHQA_ROOT),
        help="Path to the VideoMathQA dataset root",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max refinement iterations per sample")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index within the sampled subset to process (0-based); omit to process the full sampled subset",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=DEFAULT_SAMPLE_RATIO,
        help="Fraction of VideoMathQA samples to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Fixed random seed for subset sampling",
    )
    parser.add_argument(
        "--use-retrieved-context",
        action="store_true",
        help="Add retrieved_context (top-k question-relevant segments) to PREPROCESSED_ARTIFACTS",
    )
    parser.add_argument(
        "--segment-size",
        type=float,
        default=30.0,
        help="Seconds per non-overlapping segment for video_overview / dense caption cache",
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotation_file).expanduser().resolve()
    dataset_root = Path(args.videomathqa_root).expanduser().resolve()
    if not annotation_path.exists():
        raise SystemExit(f"Error: annotation file not found: {annotation_path}")
    if not dataset_root.exists():
        raise SystemExit(f"Error: VideoMathQA root not found: {dataset_root}")
    if not (0.0 < args.sample_ratio <= 1.0):
        raise SystemExit(f"Error: --sample-ratio must be in (0, 1], got {args.sample_ratio}")

    results_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else DEFAULT_RESULTS_DIR
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    data = _load_videomathqa_samples(annotation_path, dataset_root)
    if not data:
        raise SystemExit(f"Error: no samples found in {annotation_path}")

    sampled_data = _select_random_subset(
        data,
        sample_ratio=args.sample_ratio,
        sample_seed=args.seed,
    )
    if args.index is not None:
        if args.index < 0 or args.index >= len(sampled_data):
            raise SystemExit(
                f"Error: --index {args.index} out of range for sampled subset "
                f"(0..{len(sampled_data) - 1})"
            )
        sampled_data = [sampled_data[args.index]]

    print(
        f"Selected {len(sampled_data)} / {len(data)} VideoMathQA samples "
        f"({args.sample_ratio:.1%}) with seed {args.seed}"
    )

    openai_api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "").strip()
    planner_api_base = _env_list(
        "PLANNER_API_BASE", os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    )
    planner_api_keys = _env_list("PLANNER_API_KEY", os.environ.get("API_KEY", openai_api_key))
    if not planner_api_keys:
        raise SystemExit("Error: set PLANNER_API_KEY, API_KEY, or OPENAI_API_KEY before running.")

    vlm_api_base = _env_list("VLM_API_BASE")
    vlm_api_keys = _env_list("VLM_API_KEY", openai_api_key) if vlm_api_base else None
    if vlm_api_base and not vlm_api_keys:
        raise SystemExit("Error: set VLM_API_KEY, API_KEY, or OPENAI_API_KEY before running.")
    local_vlm_model_name = os.environ.get("LOCAL_VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    default_remote_vlm_model = os.environ.get("API_MODEL_NAME", "gpt-5.4")
    if vlm_api_base:
        vlm_model_name = os.environ.get("VLM_MODEL_NAME", default_remote_vlm_model)
    else:
        vlm_model_name = os.environ.get("VLM_MODEL_NAME", local_vlm_model_name)

    chart_mode = os.environ.get("CHART_MODE", "vlm")
    chart_model_name = os.environ.get("CHART_MODEL_NAME", vlm_model_name)
    planner_model_name = os.environ.get("PLANNER_MODEL_NAME", os.environ.get("API_MODEL_NAME", "gpt-5.4"))

    verifier_model_name = os.environ.get("VERIFIER_MODEL_NAME") or None
    verifier_api_base = _env_list("VERIFIER_API_BASE") or None
    verifier_api_keys = _env_list("VERIFIER_API_KEY") or None

    _tg_dev_env = os.environ.get("TEMPORAL_GROUNDER_DEVICE_INDEX", "").strip()
    temporal_grounder_device_index = int(_tg_dev_env) if _tg_dev_env.isdigit() else None

    if vlm_api_base:
        print(f"Using remote VLM API: model={vlm_model_name} base={vlm_api_base[0]}")
    else:
        print(f"Using local VLM model: {vlm_model_name}")

    demo = None
    dataset_folder = str((Path(_eval_dir) / "data").resolve())
    debug_root = str(DEFAULT_DEBUG_DIR.resolve())

    for index, item in enumerate(sampled_data, start=1):
        record = dict(item)
        video_path = str(item.get("video_path", "")).strip()
        question = str(item.get("question", "")).strip()
        options = list(item.get("options") or [])
        trace_steps = []
        input_answer = item.get("answer")
        pipeline_answer = None

        video_stem = Path(video_path).stem if video_path else "unknown"
        out_dir = results_dir / video_stem

        print(
            f"\n[{index}/{len(sampled_data)}] {Path(video_path).name or '<missing video>'} "
            f"(question_id={item.get('question_id')}, source_index={item.get('source_index')})"
        )
        if item.get("initial_trace_steps"):
            print("  Ignoring existing VideoMathQA steps; generating a fresh trace.")

        if not video_path or not os.path.exists(video_path):
            record["refiner_error"] = f"Video file not found: {video_path}"
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_result_dir(record, video_path, out_dir, args.sample_ratio, args.seed)
            print(f"  Saved: {out_dir / 'meta.json'}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_stale_split_outputs(out_dir)

        def _on_iteration_complete(iteration_record):
            n = iteration_record.get("iteration")
            if n is None:
                return
            _write_json(out_dir / f"refinement_{int(n)}.json", iteration_record)

        try:
            if demo is None:
                demo = VideoQADemo(
                    video_path=video_path,
                    question=question,
                    answer=pipeline_answer,
                    options=options,
                    dataset_folder=dataset_folder,
                    use_subtitle=False,
                    refinement_debug_root=debug_root,
                    dense_frame_fps=1.0,
                    use_clip_retrieval=False,
                    dense_segment_half_width=0.5,
                    retrieval_top_k=10,
                    dense_frame_embed_batch=8,
                    vlm_model_name=vlm_model_name,
                    vlm_api_base=vlm_api_base,
                    vlm_api_keys=vlm_api_keys,
                    planner_model_name=planner_model_name,
                    planner_api_base=planner_api_base,
                    planner_api_keys=planner_api_keys,
                    verifier_model_name=verifier_model_name,
                    verifier_api_base=verifier_api_base,
                    verifier_api_keys=verifier_api_keys,
                    chart_mode=chart_mode,
                    chart_model_name=chart_model_name,
                    temporal_grounder_device_index=temporal_grounder_device_index,
                    use_retrieved_context=args.use_retrieved_context,
                    segment_size_s=args.segment_size,
                )
            else:
                demo.load_sample(
                    video_path=video_path,
                    question=question,
                    answer=pipeline_answer,
                    options=options,
                )

            record["refiner_result"] = demo.run_refinement_pipeline(
                trace_steps=trace_steps if trace_steps else None,
                max_iterations=args.max_iterations,
                on_iteration_complete=_on_iteration_complete,
            )
            if pipeline_answer is None and input_answer:
                final_answer = record["refiner_result"].get("final_answer") or ""
                record["refiner_result"]["is_correct"] = demo._answers_match(
                    final_answer, input_answer, options=options
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            record["refiner_error"] = str(e)
            record.pop("refiner_result", None)

        _save_result_dir(record, video_path, out_dir, args.sample_ratio, args.seed)
        print(f"  Saved: {out_dir}")

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()