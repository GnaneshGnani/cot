## Preprocessing Stage (Run Once Per Video)

Before entering the iterative loop, extract reusable artifacts:

1. **ASR transcript** (with word-level timestamps) -- Whisper/WhisperX
2. **Dense captions** (segment-level descriptions) -- Qwen2.5-VL or PLLaVA
3. **Audio event timeline** -- FLAM or CLAP
4. **Keyframe index** -- K-frames or KeyScore

These are cached and reused across all QA pairs for the same video, and across verification iterations.
