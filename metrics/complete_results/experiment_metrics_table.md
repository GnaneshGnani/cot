# Experiment Metrics Summary (Complete Results)

Experiments as rows, metrics as columns. Values are from the aggregated summary (last line) of each JSONL file.

| experiment | m1_answer_faithfulness.jsonl | m2_stepwise_relevance.jsonl | m3_causal_coherence.jsonl | m4_completeness.jsonl | m5_factual_accuracy.jsonl | m6_efficiency.jsonl | m7_planning_presence.jsonl | m8_modality_coverage.jsonl | m9_temporal_grounding.jsonl |
|---|---|---|---|---|---|---|---|---|---|
| audio_steps_only | 0.6121 | 0.7180 | 0.9227 | 0.0061 | 0.5061 | 0.9593 | 0.1697 | 0.4939 | 1.0000 |
| evidence_only | 0.7394 | 0.8534 | 0.6976 | 0.0111 | 0.5000 | 0.9625 | 0.1212 | 0.5000 | 0.9915 |
| first_step_only | 0.3879 | 0.5818 | 1.0000 | 0.0100 | 0.5000 | 1.0000 | 0.1455 | 0.5000 | 1.0000 |
| inferences_only | 0.8970 | 0.7854 | 0.6674 | 0.0218 | 0.5000 | 0.9564 | 0.2121 | 0.5000 | 1.0000 |
| skip_last_step | 0.5091 | 0.8109 | 0.8303 | 0.0130 | 0.5000 | 0.9551 | 0.2545 | 0.5000 | 0.9919 |
| truncate_25 | 0.3879 | 0.5879 | 0.9879 | 0.0100 | 0.5000 | 0.9919 | 0.1455 | 0.5000 | 1.0000 |
| truncate_50 | 0.4303 | 0.6924 | 0.8997 | 0.0100 | 0.5000 | 0.9838 | 0.2121 | 0.5000 | 1.0000 |
| truncate_75 | 0.4848 | 0.7980 | 0.8268 | 0.0130 | 0.5000 | 0.9603 | 0.2485 | 0.5000 | 0.9919 |
| vision_steps_only | 0.7394 | 0.7751 | 0.8866 | 0.0090 | 0.5061 | 0.9695 | 0.1879 | 0.4939 | 0.9909 |

## Metric definitions (from summary keys)
- **m1_answer_faithfulness**: average_af_score
- **m2_stepwise_relevance**: average_srs_score
- **m3_causal_coherence**: average_ccc_score
- **m4_completeness**: average_cs_score
- **m5_factual_accuracy**: average_fas_score
- **m6_efficiency**: average_efficiency_score
- **m7_planning_presence**: fraction_with_plan
- **m8_modality_coverage**: mean of average_mc_visual and average_mc_audio
- **m9_temporal_grounding**: average_tgs_score
