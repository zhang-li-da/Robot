# g1_asap_spiderman_l2 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.984 | 111.402 | 0.317 | 19.298 | anchor_ori |
| adapted | 0.984 | 111.225 | 0.967 | 16.852 | anchor_pos |
| best_evolved | 1.000 | 112.993 | 0.712 | 19.862 | anchor_ori |

## 最佳进化候选

- genome: `gen0_m3_000`
- generation_dir: `outputs/evolution_asap/g1_asap_spiderman_l2/20260602_234929_542205_gen00`
- population_status: `success_ceiling_quality_task`
- target_met: `False`
- mean_final_yaw_error: `0.06967581732897088`
- mean_final_speed: `0.2502510459162295`
- mean_final_ang_speed: `0.6003185659646988`

## 下一代重点

- success-rate improvement target is infeasible at the current baseline; switch to harder tasks or quality/robustness metrics
- treat this as a proxy quality task, not a final stunt success-rate benchmark
