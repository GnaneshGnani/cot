import copy
import os
import sys

EXPERIMENTS = [
    "full_cot",
    "skip_last_step",
    "truncate_50",
    "truncate_25",
    "truncate_75",
    "first_step_only",
    "evidence_only",
    "inferences_only",
    "vision_steps_only",
    "audio_steps_only",
]


def get_experiment():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("EXPERIMENT", "full_cot")


def _get_modality(step):
    return str(step.get("modality") or "").lower()


def apply_experiment(sample, experiment):
    if experiment == "full_cot":
        return sample
    sample = copy.deepcopy(sample)
    steps = sample.get("reasoning_steps") or []
    if not steps:
        return sample

    if experiment == "skip_last_step":
        steps = steps[:-1]
    elif experiment == "truncate_50":
        steps = steps[: max(1, len(steps) // 2)]
    elif experiment == "truncate_25":
        steps = steps[: max(1, len(steps) // 4)]
    elif experiment == "truncate_75":
        steps = steps[: max(1, int(len(steps) * 0.75))]
    elif experiment == "first_step_only":
        steps = steps[:1]
    elif experiment == "evidence_only":
        for s in steps:
            for k in ("inference", "Inference", "infefence"):
                s[k] = ""
    elif experiment == "inferences_only":
        for s in steps:
            for k in ("evidence", "evidece", "evience", "evodence", "nevidence"):
                s[k] = ""
    elif experiment == "vision_steps_only":
        steps = [s for s in steps if _get_modality(s) == "vision"]
    elif experiment == "audio_steps_only":
        steps = [s for s in steps if _get_modality(s) == "audio"]

    sample["reasoning_steps"] = steps
    return sample
