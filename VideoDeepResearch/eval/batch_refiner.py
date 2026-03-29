#!/usr/bin/env python3
"""
Batch-run the trace refinement pipeline (refiner.VideoQADemo) over a JSONL dataset.

Run from the eval directory (or any cwd — script chdirs to eval/ for vllm_io_files).

Planner / temporal: refiner.VideoQADemo._text2text uses OpenAI HTTP when API_BASE_URL
(and temporal URL) are not localhost — e.g. gpt-5 with API_MODEL_NAME and
API_MODEL_NAME_TEMPORAL_GROUNDING (see run_refiner.slurm).

If API_BASE_URL points to localhost, start vllm_server_planner.py and
vllm_server_temporal_grounder.py with matching API_MODEL_NAME* instead.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure HF cache before heavy imports (matches generate_traces.py)
os.environ.setdefault("HF_HOME", "/fs/nexus-scratch/gnanesh/.cache/huggingface")

EVAL_DIR = Path(__file__).resolve().parent


def sample_key(video_rel: str, question: str) -> str:
    payload = json.dumps({"video": video_rel, "question": question}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def reasoning_steps_to_trace_steps(reasoning_steps: list) -> List[str]:
    if not reasoning_steps:
        return []
    lines = []
    for step in reasoning_steps:
        if not isinstance(step, dict):
            lines.append(str(step))
            continue
        modality = step.get("modality") or ""
        ev = step.get("evidence") or ""
        inf = step.get("inference") or ""
        lines.append(f"[{modality}] {ev} -> {inf}".strip())
    return lines


def load_done_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("refiner_sample_key")
                if k:
                    done.add(k)
            except json.JSONDecodeError:
                continue
    return done


def group_indices_by_video(rows: List[dict], benchmark_root: Path) -> Dict[str, List[int]]:
    """Map absolute video path -> list of row indices."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        rel = row.get("video") or ""
        if not rel:
            continue
        abs_path = (benchmark_root / rel).resolve()
        groups[str(abs_path)].append(i)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch refiner on JSONL (OmniVideoBench-style).")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/fs/nexus-scratch/gnanesh/cot/OmniVideoBench/data_short_under1min.jsonl"),
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/fs/nexus-scratch/gnanesh/cot/OmniVideoBench"),
        help="Root dir to resolve relative video= paths against.",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help="Folder for clips/embeddings (default: same as --benchmark-root).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=EVAL_DIR / "results" / "refiner_results.jsonl",
        help="Append-only output JSONL.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=EVAL_DIR / "results" / "refiner_errors.log",
        help="Append error log.",
    )
    parser.add_argument("--clip-duration", type=int, default=5)
    parser.add_argument("--use-subtitle", action="store_true", help="Load subtitles if present.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples (0 = all).")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose key exists in output.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate JSONL, paths, and grouping; do not load models.",
    )
    args = parser.parse_args()

    os.chdir(EVAL_DIR)
    sys.path.insert(0, str(EVAL_DIR.parent))

    benchmark_root = args.benchmark_root.resolve()
    dataset_folder = args.dataset_folder or str(benchmark_root)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    groups = group_indices_by_video(rows, benchmark_root)
    if args.dry_run:
        n = len(rows)
        n_videos = len(groups)
        missing = sum(
            1 for vp in groups if not os.path.isfile(vp)
        )
        print(f"dry-run: rows={n} unique_videos={n_videos} missing_video_files={missing}")
        for vp in sorted(groups.keys())[:5]:
            print(f"  video={vp} exists={os.path.isfile(vp)} n_q={len(groups[vp])}")
        if n_videos > 5:
            print(f"  ... and {n_videos - 5} more videos")
        return 0

    from refiner import VideoQADemo

    done = load_done_keys(args.output_jsonl) if args.resume else set()

    # Stable order: by first occurrence index
    video_order = sorted(groups.keys(), key=lambda vp: min(groups[vp]))

    processed = 0
    skipped = 0

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f, open(
        args.error_log, "a", encoding="utf-8"
    ) as err_f:
        for video_path in video_order:
            indices = sorted(groups[video_path])
            if not os.path.isfile(video_path):
                for idx in indices:
                    row = rows[idx]
                    k = sample_key(row.get("video", ""), row.get("question", ""))
                    err_f.write(f"missing_video\t{k}\t{video_path}\n")
                    err_f.flush()
                continue

            demo: Optional["VideoQADemo"] = None
            for idx in indices:
                if args.limit and processed >= args.limit:
                    break
                row = rows[idx]
                question = row.get("question") or ""
                rel_video = row.get("video") or ""
                k = sample_key(rel_video, question)
                if args.resume and k in done:
                    skipped += 1
                    continue

                trace_steps = reasoning_steps_to_trace_steps(row.get("reasoning_steps") or [])
                options = row.get("options")
                if options is None:
                    options = []
                answer = row.get("answer")

                try:
                    if demo is None:
                        demo = VideoQADemo(
                            video_path=video_path,
                            question=question,
                            answer=answer,
                            options=options,
                            dataset_folder=dataset_folder,
                            clip_duration=args.clip_duration,
                            use_subtitle=args.use_subtitle,
                        )
                    else:
                        demo.set_task(question, answer, options)

                    result = demo.run_refinement_pipeline(trace_steps)

                    record = {
                        "refiner_sample_key": k,
                        "video": rel_video,
                        "video_abs": video_path,
                        "question": question,
                        "answer": answer,
                        "options": options,
                        "question_type": row.get("question_type"),
                        "correct_option": row.get("correct_option"),
                        "is_correct": row.get("is_correct"),
                        "initial_trace": result.get("initial_trace"),
                        "initial_answer": result.get("initial_answer"),
                        "final_trace": result.get("final_trace"),
                        "final_answer": result.get("final_answer"),
                        "verifier_raw": result.get("verifier_raw"),
                        "verifier_output": result.get("verifier_output"),
                        "iteration_history": result.get("iteration_history"),
                        "all_iterations": result.get("all_iterations"),
                        "max_iterations": result.get("max_iterations"),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    done.add(k)
                    processed += 1
                except Exception as e:
                    err_f.write(
                        f"{k}\t{video_path}\t{question[:80]!r}\t{type(e).__name__}: {e}\n"
                    )
                    err_f.write(traceback.format_exc() + "\n")
                    err_f.flush()
                    demo = None

            if args.limit and processed >= args.limit:
                break

    print(f"Done. processed={processed} skipped_resume={skipped} output={args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
