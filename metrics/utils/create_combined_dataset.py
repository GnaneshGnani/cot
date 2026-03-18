"""Create a combined JSONL with 10 OmniVideoBench + 10 VideoMathQA samples, absolute video paths."""
import json
import os

PROJECT_ROOT = "/fs/nexus-scratch/gnanesh/cot"
OVB_DIR = os.path.join(PROJECT_ROOT, "OmniVideoBench")
VMQA_DIR = os.path.join(PROJECT_ROOT, "VideoMathQA")


def _ovb_sample(obj):
    """Convert OmniVideoBench sample to unified format with absolute video path."""
    vid = obj.get("video", "")
    if vid and not os.path.isabs(vid):
        abs_path = os.path.join(OVB_DIR, vid)
    else:
        abs_path = vid
    out = {
        "video": os.path.abspath(abs_path) if abs_path else "",
        "question": obj.get("question", ""),
        "options": obj.get("options", []),
        "answer": obj.get("answer", ""),
        "correct_option": obj.get("correct_option", obj.get("answer", "")),
        "reasoning_steps": obj.get("reasoning_steps", []),
        "duration": obj.get("duration"),
    }
    for k in ("question_type", "video_type", "audio_type"):
        if k in obj:
            out[k] = obj[k]
    return out


def _parse_vmqa_steps(steps_str):
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


def _vmqa_sample(obj):
    """Convert VideoMathQA sample to unified format with absolute video path."""
    vid = obj.get("videoID", "")
    if vid:
        abs_path = os.path.join(VMQA_DIR, "videos", f"{vid}.mp4")
    else:
        abs_path = ""
    return {
        "video": os.path.abspath(abs_path) if abs_path else "",
        "question": obj.get("question", ""),
        "options": obj.get("options", []),
        "answer": obj.get("answer", ""),
        "correct_option": obj.get("answer", ""),
        "reasoning_steps": _parse_vmqa_steps(obj.get("steps", "")),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.environ.get("OUTPUT_PATH") or os.path.join(
        script_dir, "results", "combined_20", "combined_20_samples.jsonl"
    )
    n_ovb = int(os.environ.get("N_OVB", "10"))
    n_vmqa = int(os.environ.get("N_VMQA", "10"))

    samples = []

    ovb_file = os.path.join(OVB_DIR, "data_short_under1min.jsonl")
    if os.path.exists(ovb_file):
        with open(ovb_file) as f:
            for i, line in enumerate(f):
                if i >= n_ovb:
                    break
                if line.strip():
                    samples.append(_ovb_sample(json.loads(line)))
    else:
        ovb_json = os.path.join(OVB_DIR, "data.json")
        if os.path.exists(ovb_json):
            with open(ovb_json) as f:
                data = json.load(f)
            for obj in data[:n_ovb]:
                samples.append(_ovb_sample(obj))

    vmqa_file = os.path.join(VMQA_DIR, "videomathqa_mcq_test.json")
    if os.path.exists(vmqa_file):
        with open(vmqa_file) as f:
            data = json.load(f)
        for obj in data[:n_vmqa]:
            samples.append(_vmqa_sample(obj))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as out:
        for s in samples:
            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Created {output_path} with {len(samples)} samples ({n_ovb} OmniVideoBench, {n_vmqa} VideoMathQA)")


if __name__ == "__main__":
    main()
