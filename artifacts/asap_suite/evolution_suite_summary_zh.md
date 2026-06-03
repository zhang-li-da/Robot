# ASAP 特技动作自主进化 Suite 汇总

- 任务数：`5`
- 已有最终复评任务：`5`
- 仅有中期结果任务：`0`
- 达到 >8% 成功率提升任务：`5`

## 结果表

| 任务 | 状态 | 对比候选 | baseline成功率 | adapted成功率 | 进化成功率 | 提升 | 达标 | 主要终止 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| g1_asap_turn_jump_l4 | final | final_eval | 0.422 | 0.000 | 0.578 | 0.156 | yes | anchor_pos |
| g1_asap_squat_l3_lowposture | final | final_eval | 0.359 | 0.422 | 0.828 | 0.469 | yes | anchor_pos |
| g1_asap_jump_forward_l4 | final | final_eval | 0.047 | 0.000 | 0.906 | 0.859 | yes | none |
| g1_asap_side_jump_l4 | final | final_eval | 0.000 | 0.000 | 0.969 | 0.969 | yes | none |
| g1_asap_cr7_l2_dynamic | final | final_eval | 0.000 | 0.000 | 0.984 | 0.984 | yes | none |

## 说明

- `interim` 表示当前只有 stage1/小预算结果，不能作为最终考核结论。
- `final` 表示已经生成不少于 50 次任务执行的正式复评摘要。
- 最终考核以 `final_eval` 的成功率提升为准；没有 final 结果时只作为调试趋势。
