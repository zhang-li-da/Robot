# g1_asap_squat_l3_lowposture 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.359 | 47.463 | 0.072 | 15.514 | ee_body_pos |
| adapted | 0.422 | 53.211 | 0.216 | 19.507 | ee_body_pos |
| best_evolved | 1.000 | 112.532 | 0.274 | 21.263 | ee_body_pos |
| final_eval | 0.828 | 95.228 | 0.246 | 21.116 | anchor_pos |

## 最佳进化候选

- genome: `gen2_m3_001`
- generation_dir: `outputs/evolution_asap/g1_asap_squat_l3_lowposture/20260603_165940_426435_gen02`
- population_status: `target_met`
- target_met: `True`
- mean_final_yaw_error: `0.3731559934094548`
- mean_final_speed: `0.31938036589417607`
- mean_final_ang_speed: `0.7264936259016395`

## 下一代重点

- use comparison evals as ablation evidence and require candidates to beat baseline under the final protocol
- preserve recovery-friendly sampling and avoid stricter early termination in the next generation
- avoid stricter anchor_pos termination; tune task rewards only after anchor tracking stabilizes

## 最终复评

- final_eval_label: `gen2_m3_001_repair_final64`
- final_eval_path: `artifacts/g1_asap_squat_l3_lowposture/eval/best_evolved_repair_gen2_m3_001_64ep.json`
- success_rate_delta_vs_baseline: `0.46875`
- minimum_trials_met: `True`
- target_improvement_met: `True`
- target_met: `True`
