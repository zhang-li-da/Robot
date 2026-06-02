# g1_jump_leap 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.000 | -2.060 | 0.861 | 0.561 | anchor_pos |
| adapted | 0.000 | 1.801 | 1.671 | 5.425 | ee_body_pos |
| best_evolved | 0.625 | 72.262 | 5.087 | 9.333 | ee_body_pos |
| final_eval | 0.547 | 64.651 | 5.006 | 9.705 | anchor_pos |

## 最佳进化候选

- genome: `gen0_m3_001`
- generation_dir: `outputs/evolution_amass/jump_leap/20260602_163404_197448_gen00`
- population_status: `target_met`
- target_met: `True`

## 下一代重点

- differentiate legal support contact from ee/body tracking failure

## 最终复评

- final_eval_label: `gen0_m3_001_final64`
- final_eval_path: `artifacts/g1_jump_leap/eval/best_evolved_64ep.json`
- success_rate_delta_vs_baseline: `0.546875`
- minimum_trials_met: `True`
- target_improvement_met: `True`
- target_met: `True`
