#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import traceback
from contextlib import contextmanager

_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import hf_cache

hf_cache.ensure_hf_cache_env()

import torch
import whisperx


@contextmanager
def _whisperx_torch_load_compat():
    original_torch_load = getattr(torch, "load", None)
    if original_torch_load is None:
        yield
        return

    def _compat_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _dedupe_paths(paths):
    deduped = []
    seen = set()
    for path in paths:
        raw = str(path or "").strip()
        if not raw:
            continue
        key = os.path.realpath(raw)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(raw)
    return deduped


def _sanitize_ld_library_path():
    current = os.environ.get("LD_LIBRARY_PATH", "")
    gpu_paths = []
    for name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            mod = importlib.import_module(name)
            gpu_paths.extend(list(getattr(mod, "__path__", []) or []))
        except Exception:
            continue

    keep = []
    for entry in current.split(":"):
        raw = entry.strip()
        if not raw:
            continue
        normalized = os.path.realpath(raw).lower()
        if "/site-packages/nvidia/" in normalized:
            keep.append(raw)
            continue
        if any(
            marker in normalized
            for marker in (
                "/miniconda",
                "/anaconda",
                "/miniforge",
                "/mambaforge",
                "/micromamba",
                "/conda/",
            )
        ):
            continue
        keep.append(raw)

    keep = _dedupe_paths(list(gpu_paths) + keep)
    if keep:
        os.environ["LD_LIBRARY_PATH"] = ":".join(keep)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)


def _transcribe_window(
    video_path: str,
    start_time: float,
    end_time: float,
    model_name: str,
    device: str,
    aux_device: str,
    compute_type: str,
    batch_size: int,
    language: str | None,
):
    _sanitize_ld_library_path()
    transcribe_device_obj = torch.device(device)
    transcribe_device = transcribe_device_obj.type
    transcribe_device_index = int(transcribe_device_obj.index or 0)
    aux_device = str(torch.device(aux_device))

    with _whisperx_torch_load_compat():
        vad_model = whisperx.asr.load_vad_model(torch.device(aux_device))

    with _whisperx_torch_load_compat():
        model = whisperx.load_model(
            model_name,
            transcribe_device,
            device_index=transcribe_device_index,
            compute_type=compute_type,
            vad_model=vad_model,
        )

    audio = whisperx.load_audio(video_path)
    sample_rate = float(getattr(getattr(whisperx, "audio", None), "SAMPLE_RATE", 16000))
    sample_start = max(0, int(float(start_time) * sample_rate))
    sample_end = min(len(audio), int(float(end_time) * sample_rate))
    audio_window = audio[sample_start:sample_end] if sample_end > sample_start else audio[0:0]

    if len(audio_window) == 0:
        return {
            "language_detected": "unknown",
            "transcript": "",
            "full_transcript": "",
            "segments": [],
            "words": [],
            "asr_backend": "whisperx_sidecar",
            "asr_runner": "sidecar",
            "asr_device": str(transcribe_device_obj),
            "asr_device_index": transcribe_device_index,
            "asr_aux_device": aux_device,
            "asr_compute_type": compute_type,
            "whisperx_model": model_name,
        }

    transcribe_kwargs = {"batch_size": int(batch_size)}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_window, **transcribe_kwargs)
    lang = result.get("language") or "en"
    align_warning = None

    try:
        with _whisperx_torch_load_compat():
            model_a, meta = whisperx.load_align_model(language_code=lang, device=aux_device)
        aligned = whisperx.align(
            result["segments"],
            model_a,
            meta,
            audio_window,
            aux_device,
            return_char_alignments=False,
        )
        segs = aligned.get("segments", [])
    except Exception as ex:
        align_warning = f"{type(ex).__name__}: {ex}"
        segs = [
            {
                "start": float(s.get("start", 0)),
                "end": float(s.get("end", 0)),
                "text": str(s.get("text", "")),
                "score": 0.75,
                "words": [],
            }
            for s in result.get("segments", [])
        ]

    offset = float(start_time)
    normalized = []
    for seg in segs:
        item = dict(seg)
        item["start"] = float(item.get("start", 0)) + offset
        item["end"] = float(item.get("end", 0)) + offset
        words = []
        for word in item.get("words") or []:
            word_item = dict(word)
            if "start" in word_item:
                word_item["start"] = float(word_item.get("start", 0)) + offset
            if "end" in word_item:
                word_item["end"] = float(word_item.get("end", 0)) + offset
            words.append(word_item)
        item["words"] = words
        normalized.append(item)

    segments = []
    words_flat = []
    for seg in normalized:
        segments.append(
            {
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": str(seg.get("text", "")).strip(),
                "speaker": seg.get("speaker"),
                "confidence": float(seg.get("score", 0.9) or 0.9),
            }
        )
        for word in seg.get("words") or []:
            words_flat.append(
                {
                    "word": str(word.get("word", "")),
                    "start": float(word.get("start", 0)),
                    "end": float(word.get("end", 0)),
                    "confidence": float(word.get("score", 0.0) or 0.0),
                }
            )

    transcript = " ".join(seg["text"] for seg in segments).strip()
    payload = {
        "language_detected": lang,
        "transcript": transcript,
        "full_transcript": transcript,
        "segments": segments,
        "words": words_flat,
        "asr_backend": "whisperx_sidecar",
        "asr_runner": "sidecar",
        "asr_device": str(transcribe_device_obj),
        "asr_device_index": transcribe_device_index,
        "asr_aux_device": aux_device,
        "asr_compute_type": compute_type,
        "whisperx_model": model_name,
    }
    if align_warning:
        payload["align_warning"] = align_warning
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--model-name", default=os.getenv("WHISPERX_MODEL", "small"))
    parser.add_argument("--device", default=os.getenv("WHISPERX_DEVICE", "cuda:0"))
    parser.add_argument("--aux-device", default=os.getenv("WHISPERX_AUX_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.getenv("WHISPERX_COMPUTE_TYPE", "float16"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("WHISPERX_BATCH", "8")))
    parser.add_argument("--language", default=os.getenv("WHISPERX_LANGUAGE", "").strip() or None)
    args = parser.parse_args()

    try:
        payload = _transcribe_window(
            video_path=args.video_path,
            start_time=float(args.start_time),
            end_time=float(args.end_time),
            model_name=str(args.model_name),
            device=str(args.device),
            aux_device=str(args.aux_device),
            compute_type=str(args.compute_type),
            batch_size=int(args.batch_size),
            language=args.language,
        )
    except Exception as ex:
        print(
            json.dumps(
                {
                    "error_type": type(ex).__name__,
                    "error": str(ex),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
