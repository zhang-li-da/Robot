# ASAP 特技动作自主进化 Suite 汇总

- 任务数：`4`
- 已有最终复评任务：`0`
- 仅有中期结果任务：`1`
- 达到 >8% 成功率提升任务：`1`

## 结果表

| 任务 | 状态 | 对比候选 | baseline成功率 | adapted成功率 | 进化成功率 | 提升 | 达标 | 主要终止 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| g1_asap_jump_forward_l5 | interim | best_evolved_stage1 | 0.000 | 0.000 | 0.125 | 0.125 | yes | anchor_ori |
| g1_asap_turn_jump_l5 | missing | - | - | - | - | - | - | - |
| g1_asap_spiderman_l2 | missing | - | - | - | - | - | - | - |
| g1_asap_single_foot_jump_l2 | missing | - | - | - | - | - | - | - |

## 说明

- `interim` 表示当前只有 stage1/小预算结果，不能作为最终考核结论。
- `final` 表示已经生成不少于 50 次任务执行的正式复评摘要。
- 最终考核以 `final_eval` 的成功率提升为准；没有 final 结果时只作为调试趋势。
