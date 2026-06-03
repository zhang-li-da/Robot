# g1_asap_side_jump_l4 自主进化实验摘要

## 对比结果

| 方法 | 成功率 | Fitness | 平均位移 | 平均回报 | 主要终止 |
| --- | ---: | ---: | ---: | ---: | --- |
| BeyondMimic baseline | 0.000 | 9.226 | 0.860 | 13.597 | none |
| adapted_task_rewards | 0.000 | 8.133 | 0.690 | 13.547 | none |
| 旧 best_evolved final64 | 0.094 | 19.883 | 1.058 | 13.947 | none |
| gen6_baseline_000 final64 | 0.922 | 103.915 | 1.516 | 13.771 | none |
| gen6_m3_000 final64 | 0.969 | 108.609 | 1.556 | 13.887 | none |

## 最佳进化候选

- genome: `gen6_m3_000`
- generation_dir: `outputs/evolution_asap/g1_asap_side_jump_l4/20260603_191414_393620_gen06`
- stage2_eval: `32/32`
- final64_eval: `62/64`
- final64_eval_path: `artifacts/g1_asap_side_jump_l4/eval/best_evolved_gen6_m3_000_64ep.json`
- checkpoint: `/root/whole_body_tracking-main/logs/rsl_rl/g1_asap_side_jump_l4/2026-06-03_20-11-41_evo_gen6_m3_000_stage2/model_998.pt`

## 关键修复

- 修复 JumpLeap 任务中 `task_progress.params.target_x` 仍使用默认 `5.0` 的问题，改为从任务配置绑定 `target_x=1.25`。
- 对 side-jump 位移短板加入 velocity shaping 与 stagnation penalty，避免策略用原地稳定存活替代侧向/前向位移。
- 在 LLM 候选归一化路径中加入 JumpLeap progress guard，使 M3 生成候选自动继承任务目标、采样和终止容忍约束。

## 最终复评

- baseline_success_rate: `0.000`
- final_success_rate: `0.969`
- success_rate_delta_vs_baseline: `0.969`
- required_delta: `0.080`
- minimum_trials_met: `True`
- target_improvement_met: `True`
- target_met: `True`

