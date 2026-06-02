# g1_asap_jump_forward_l5 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.000 | 1.369 | 0.057 | 6.063 | anchor_pos |
| adapted | 0.000 | 2.984 | 0.323 | 8.650 | ee_body_pos |
| best_evolved | 0.125 | 23.502 | 1.417 | 14.114 | anchor_ori |

## 最佳进化候选

- genome: `gen0_m3_000`
- generation_dir: `outputs/evolution_asap/g1_asap_jump_forward_l5/20260602_185815_100995_gen00`
- population_status: `target_met`
- target_met: `True`

## 下一代重点

- increase behavior diversity through entropy, phase sampling, or warm-start curriculum
