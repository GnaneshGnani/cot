"""Convert VideoMathQA format to the metrics pipeline format (jsonl with reasoning_steps)."""
import json
import os


def parse_steps(steps_str):
    """Convert VideoMathQA 'steps' JSON string to reasoning_steps list."""
    if not steps_str:
        return []
    try:
        raw = json.loads(steps_str) if isinstance(steps_str, str) else steps_str
    except (json.JSONDecodeError, TypeError):
        return []
    steps = []
    for i in sorted(raw.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
        text = raw.get(str(i), raw.get(i, ""))
        if text:
            steps.append({"evidence": "", "inference": str(text)})
    return steps


def convert_sample(obj, videos_subdir="videos"):
    """Convert one VideoMathQA sample to metrics format."""
    vid = obj.get("videoID", "")
    return {
        "video": f"{videos_subdir}/{vid}.mp4" if vid else "",
        "question": obj.get("question", ""),
        "options": obj.get("options") or [],
        "answer": obj.get("answer", ""),
        "correct_option": obj.get("answer", ""),
        "reasoning_steps": parse_steps(obj.get("steps", "")),
    }


def load_videomathqa(path):
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main():
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    vmqa_dir = os.environ.get("VIDEOMATHQA_DIR", os.path.join(project_root, "VideoMathQA"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment = os.environ.get("EXPERIMENT", "videomathqa")
    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass
    output_path = os.environ.get("DATA_PATH") or os.path.join(
        script_dir, "results", experiment, "videomathqa_converted.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen = set()
    samples = []
    for name in ("videomathqa_mcq_test.json", "videomathqa_mbin_test.json", "annotations.json"):
        path = os.path.join(vmqa_dir, name)
        if not os.path.exists(path):
            continue
        for obj in load_videomathqa(path):
            if max_samples is not None and len(samples) >= max_samples:
                break
            key = (obj.get("videoID", ""), obj.get("question", ""))
            if key in seen:
                continue
            seen.add(key)
            samples.append(convert_sample(obj))
        if max_samples is not None and len(samples) >= max_samples:
            break

    with open(output_path, "w") as out:
        for s in samples:
            out.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Converted {len(samples)} samples to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
