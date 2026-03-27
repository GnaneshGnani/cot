## Benchmark-Specific Adaptations

Different benchmarks have different trace formats, so a **format adapter** layer is needed:

- **OmniVideoBench:** Trace = list of (Modality, Evidence, Inference) triples. Ensure refined trace preserves this 3-tuple structure and correct V/A modality tags.
- **VideoMathQA:** Trace = numbered math solution steps. Refinement must respect mathematical notation and logical dependencies between steps. OCR and Dense Captioner are critical (equations on screen).
- **VideoEspresso:** Trace = CoT text + core frames + bboxes + temporal alignment. Refinement must update both text AND spatial/temporal grounding annotations.
- **Minerva:** Trace = free-form prose with inline timestamps (~4 per trace). Refinement must fix timestamp references within prose while maintaining readability.