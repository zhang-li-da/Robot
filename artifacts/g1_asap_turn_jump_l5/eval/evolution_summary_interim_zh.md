# g1_asap_turn_jump_l5 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.000 | 0.841 | 0.094 | 6.926 | anchor_pos |
| adapted | 0.000 | 1.070 | 0.102 | 9.488 | anchor_pos |
| best_evolved | 0.000 | 12.795 | 0.867 | 15.906 | anchor_ori |

## 最佳进化候选

- genome: `gen0_m3_002`
- generation_dir: `outputs/evolution_asap/g1_asap_turn_jump_l5/20260602_221209_464195_gen01`
- population_status: `needs_iteration`
- target_met: `False`
- mean_final_yaw_error: `1.3302924111485481`
- mean_final_speed: `0.11340846167877316`
- mean_final_ang_speed: `0.4953630901873112`

## 下一代重点

- increase yaw recovery pressure
- preserve the progress/apex/landing gains of the current best candidate
- do not relax the final yaw criterion in formal evaluation

## 任务特异诊断

- type: `aerial_turn_yaw_repair`
- failure_tags: `deterministic_collapse, low_behavior_diversity, no_success, yaw_recovery_failure`
- progress_ratio: `1.2391319179109166`
- mean_final_yaw_error: `1.3302924111485481`
