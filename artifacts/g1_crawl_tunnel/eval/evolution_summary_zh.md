# g1_crawl_tunnel 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.000 | -0.054 | 0.105 | 8.354 | ee_body_pos |
| adapted | 0.000 | -0.569 | 0.108 | 7.109 | ee_body_pos |
| best_evolved | 0.000 | 9.996 | 0.947 | 18.896 | time_out |

## 最佳进化候选

- genome: `gen1_m3_000`
- generation_dir: `outputs/evolution_amass/crawl_tunnel/20260602_153156_521231_gen01`
- population_status: `needs_iteration`
- target_met: `False`

## 下一代重点

- increase behavior diversity through entropy, phase sampling, or warm-start curriculum
