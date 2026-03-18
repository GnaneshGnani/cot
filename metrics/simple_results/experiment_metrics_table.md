# Experiment Metrics Summary

Experiments as rows, metrics as columns. Values are from the aggregated summary (last line) of each JSONL file.

| experiment | m1_answer_faithfulness.jsonl | m2_stepwise_relevance.jsonl | m3_causal_coherence.jsonl | m6_efficiency.jsonl | m7_planning_presence.jsonl | m8_modality_coverage.jsonl | m9_temporal_grounding.jsonl |
|---|---|---|---|---|---|---|---|
| audio_steps_only | 0.5818 | 0.7576 | 0.9591 | 0.9593 | 0.1697 | 0.4939 | 1.0000 |
| evidence_only | 0.7273 | 0.8221 | 0.7251 | 0.9625 | 0.1212 | 0.5000 | 0.9915 |
| first_step_only | 0.3091 | 0.6545 | 1.0000 | 1.0000 | 0.1455 | 0.5000 | 1.0000 |
| full_cot | 1.0000 | 0.9060 | 0.8510 | 0.9672 | 0.2848 | 0.5000 | 0.9915 |
| inferences_only | 0.8970 | 0.6919 | 0.6458 | 0.9564 | 0.2121 | 0.5000 | 1.0000 |
| skip_last_step | 0.5030 | 0.8091 | 0.8728 | 0.9551 | 0.2545 | 0.5000 | 0.9919 |
| truncate_25 | 0.3091 | 0.6515 | 0.9924 | 0.9919 | 0.1455 | 0.5000 | 1.0000 |
| truncate_50 | 0.3879 | 0.6944 | 0.9477 | 0.9838 | 0.2121 | 0.5000 | 1.0000 |
| truncate_75 | 0.4848 | 0.8000 | 0.8804 | 0.9603 | 0.2485 | 0.5000 | 0.9919 |
| vision_steps_only | 0.7030 | 0.7255 | 0.9087 | 0.9695 | 0.1879 | 0.4939 | 0.9909 |

## Metric definitions (from summary keys)
- **m1_answer_faithfulness**: average_af_score (answer correctness)
- **m2_stepwise_relevance**: average_srs_score
- **m3_causal_coherence**: average_ccc_score
- **m6_efficiency**: average_efficiency_score
- **m7_planning_presence**: fraction_with_plan
- **m8_modality_coverage**: mean of average_mc_visual and average_mc_audio
- **m9_temporal_grounding**: average_tgs_score
