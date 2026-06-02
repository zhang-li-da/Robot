# BeyondMimic 任务自适应算法自动进化框架

该目录保存“LLM + 进化策略”的 V1 框架，用于围绕 G1 膝爬、400m 障碍翻墙、钻洞等任务自动搜索 BeyondMimic 的任务适配参数组合。

## 设计原则

- Mimimax M3 只生成结构化 JSON 算法基因，不直接修改任意 Python 源码。
- 本地 validator 检查 schema、数值范围、任务不变量和安全边界。
- planner 将合法 genome 渲染成训练命令、评估命令和资源计划。
- scoreboard 读取评估 JSON，计算 fitness 并记录淘汰依据。
- API key 只从环境变量或 `/root/Mimimax` 读取，绝不提交到 Git。
- ASAP 数据根目录优先读取 `ASAP_DATASET_ROOT`，未设置时自动回退到 `/root/ASAP-main` 和 `/root/shared-nvme/datasets/ASAP-main`。

## 文件结构

```text
evolution/
  configs/g1_knee_climb_v1.json
  prompts/mimimax_m3_candidate_generation_zh.md
scripts/evolution/
  minimax_client.py
  schemas.py
  validator.py
  genome_ops.py
  planner.py
  scoreboard.py
  feedback_analyzer.py
  run_generation.py
  execute_generation.py
  closed_loop.py
scripts/asap_g1_task_suite.py
scripts/index_asap_motion_catalog.py
scripts/index_asap_assets.py
scripts/extract_asap_algorithm_priors.py
scripts/select_asap_evolution_tasks.py
scripts/create_asap_evolution_configs.py
scripts/create_asap_task_profiles.py
scripts/build_asap_task_adaptive_roadmap.py
scripts/run_asap_g1_evolution_experiments.sh
docs/beyondmimic_task_adaptive_evolution.md
evolution/action_catalog/stunt_motion_sources_zh.md
evolution/action_catalog/asap_motion_catalog.json
evolution/action_catalog/asap_evolution_candidate_queue.json
evolution/action_catalog/asap_evolution_candidate_queue_zh.md
evolution/action_catalog/asap_task_adaptive_roadmap.json
evolution/action_catalog/asap_task_adaptive_roadmap_zh.md
evolution/action_catalog/asap_asset_manifest.json
evolution/action_catalog/asap_asset_manifest_zh.md
evolution/algorithm_priors/asap_algorithm_priors.json
evolution/algorithm_priors/asap_algorithm_priors_zh.md
evolution/task_profiles/*.json
evolution/task_feature_schema.json
evolution/algorithm_patch_schema.json
evolution/examples/crawl_ceiling_zone_reward_v1.json
```

## ASAP 算法先验层

`scripts/extract_asap_algorithm_priors.py` 会从 `/root/ASAP-main` 中提取 ASAP 的 motion tracking reward、history observation、domain randomization、delta-action sim2real 和 PPO 配置，输出给 Mimimax M3 使用的结构化先验：

```text
evolution/algorithm_priors/asap_algorithm_priors.json
evolution/algorithm_priors/asap_algorithm_priors_zh.md
```

这些先验只作为搜索约束和迁移参考，不作为当前任务成功证据。对于 ASAP 包中没有真实后空翻、翻墙或钻洞动作的情况，框架会把 `single_foot_jump`、`jump_degree`、`squat`、`SpiderMan` 等动作标记为 proxy/pretraining，最终结论仍必须使用真实目标动作、目标碰撞几何和不少于 50 次 motion-start 评估。

## 本地密钥配置

推荐使用环境变量：

```bash
export MINIMAX_API_URL="https://api.minimax.io/v1/text/chatcompletion_v2"
export MINIMAX_API_KEY="你的 key"
export MINIMAX_MODEL="MiniMax-M3"
```

如果不设置环境变量，脚本会尝试读取 `/root/Mimimax`。脚本只打印 API URL、接口模式和模型名，不打印 key 或 key 前后缀。

注意：MiniMax 官方 M3 示例使用标准 `chatcompletion_v2` 接口；如果 `/root/Mimimax` 中仍是 `/anthropic` URL，客户端在 `api_mode=auto` 且模型为 `MiniMax-M3` 时会自动切换到同域名的 `/v1/text/chatcompletion_v2`。

## Dry-run

不调用 LLM、不启动 IsaacLab 训练，只生成和验证候选：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --dry_run \
  --population_size 4 \
  --generation 0
```

输出位于：

```text
outputs/evolution/<run_id>/
```

## 使用 Mimimax M3 生成候选

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --use_llm \
  --dry_run \
  --population_size 4 \
  --generation 0 \
  --llm_timeout 300
```

默认仍只生成计划，不启动大规模训练。正式训练前先检查输出目录中的：

```text
genomes/*.json
plans/*.json
train_commands.sh
eval_commands.sh
```

## 后续正式闭环

1. 用 `--dry_run` 检查搜索空间和命令。
2. 用 `--use_llm` 生成候选。
3. 按 `train_commands.sh` 对候选进行小预算训练。
4. 按 `eval_commands.sh` 输出评估 JSON。
5. 运行下一代时传入历史 scoreboard，让 LLM 和进化算子基于失败原因继续改进。

V1 默认先服务 G1 50cm 膝爬。翻墙和钻洞数据到位后，应新增对应 config，并扩展 task metrics。

## G1 膝爬 V1 正式执行模板

生成第一代候选：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --use_llm \
  --dry_run \
  --population_size 6 \
  --generation 0 \
  --llm_timeout 300
```

若 Mimimax 超时，脚本会回退到本地 seed 候选，仍然输出完整计划。进入输出目录后执行小预算训练：

```bash
cd /root/whole_body_tracking-main
bash outputs/evolution/<run_id>/train_commands.sh
```

训练完成后执行评估：

```bash
cd /root/whole_body_tracking-main
bash outputs/evolution/<run_id>/eval_commands.sh
```

也可以使用执行器自动做 preflight、训练、评估和 scoreboard：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/execute_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --output_dir outputs/evolution/<run_id> \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep
```

如果当前服务器的 NVIDIA Vulkan/Isaac 图形栈不可用，执行器会写出 `blocked_environment.json`，不会把候选算法标记为失败。

已有评估 JSON 可以用 `scoreboard.py` 的函数读取；下一步可把 `scoreboard.json` 传给 `--history`，让 Mimimax M3 根据失败类型继续生成第二代候选。

生成排序结果：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/scoreboard.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --output_dir outputs/evolution/<run_id> \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep
```

## 结构化反馈与闭环进化

单次候选训练结束后，先把评估结果转成 LLM 可消费的失败诊断：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/feedback_analyzer.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --output_dir outputs/evolution/<run_id> \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep
```

输出：

```text
outputs/evolution/<run_id>/feedback.json
```

`feedback.json` 会记录：

- `failure_tags`：例如 `ee_body_pos_dominant`、`early_progress_failure`、`insufficient_clearance`。
- `runtime_failures`：例如 `runtime_sigbus`、`tensorboard_writer_failure`、`runtime_gpu_memory_pressure`。
- `hypotheses`：候选失败的算法原因假设。
- `suggested_levers`：下一代应修改的 reward、termination、sampling 或 PPO 方向。
- `next_generation_focus`：传给 Mimimax M3 的硬约束改进重点。

下一代生成时同时传入历史分数和结构化反馈：

```bash
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --use_llm \
  --dry_run \
  --population_size 6 \
  --generation 1 \
  --history outputs/evolution/<run_id>/scoreboard.json \
  --feedback outputs/evolution/<run_id>/feedback.json \
  --llm_timeout 300
```

也可以直接使用闭环编排脚本，自动串联生成、执行、评分和反馈：

```bash
python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep \
  --use_llm \
  --generations 3 \
  --population_size 4 \
  --stop_on_target
```

当 baseline 已经接近满分且 `feedback_analyzer.py` 标记 `success_ceiling_quality_task` 时，`closed_loop.py` 默认停止该任务后续代数，把 GPU 资源转给更难动作或最终质量复评。若确实要继续优化 proxy 任务的质量指标，可以显式加：

```bash
python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_asap_spiderman_l2_v1.json \
  --baseline_eval artifacts/g1_asap_spiderman_l2/eval/baseline_beyondmimic.json \
  --baseline_id baseline_beyondmimic \
  --use_llm \
  --continue_on_success_ceiling
```

最终报告不要只使用 stage1 小预算评估。候选选定后应追加不少于 50 回合复评，并通过 `--final_eval` 写入摘要：

```bash
python scripts/evolution/summarize_task_evolution.py \
  --config evolution/configs/g1_jump_leap_v1.json \
  --evolution_root outputs/evolution_amass/jump_leap \
  --baseline_eval artifacts/g1_jump_leap/eval/baseline_beyondmimic.json \
  --adapted_eval artifacts/g1_jump_leap/eval/adapted_task_rewards.json \
  --final_eval artifacts/g1_jump_leap/eval/best_evolved_64ep.json \
  --final_label gen0_m3_001_final64 \
  --output_json artifacts/g1_jump_leap/eval/evolution_summary.json \
  --output_md artifacts/g1_jump_leap/eval/evolution_summary_zh.md
```

ASAP finalizer 会在复评和摘要后自动渲染 best evolved policy 的视频：

```text
artifacts/<task>/eval/best_evolved_64ep.json
artifacts/<task>/eval/evolution_summary_zh.md
artifacts/<task>/video/best_evolved_<genome_id>/rl-video-step-0.mp4
artifacts/<task>/video/best_evolved_<genome_id>/play_metrics.json
artifacts/<task>/video/best_evolved_video_manifest.json
```

对应入口：

```bash
bash scripts/finalize_asap_evolution_results.sh
```

ASAP 动作包更新后，先刷新动作目录和候选队列：

```bash
cd /root/whole_body_tracking-main
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210
python scripts/index_asap_motion_catalog.py
python scripts/extract_asap_algorithm_priors.py
python scripts/index_asap_assets.py
python scripts/create_asap_evolution_configs.py
python scripts/create_asap_task_profiles.py
python scripts/select_asap_evolution_tasks.py --limit 24
python scripts/build_asap_task_adaptive_roadmap.py --limit 18
```

候选队列输出：

```text
evolution/action_catalog/asap_evolution_candidate_queue.json
evolution/action_catalog/asap_evolution_candidate_queue_zh.md
evolution/action_catalog/asap_task_adaptive_roadmap.json
evolution/action_catalog/asap_task_adaptive_roadmap_zh.md
```

该队列按动作标签、位移、高度变化、时长和是否已有正式配置排序，并给出推荐的 `base_config`、`isaac_task`、`success_type` 和 reward 进化重点。它用于把后续新增的翻墙、钻洞、后空翻、登墙转身 motion 自动转成任务优先级和 LLM prompt 证据。`proxy` 类动作只能作为预训练或压力测试，不能作为真实特技任务的最终成功证据。

`asap_task_adaptive_roadmap_zh.md` 进一步把动作划分为正式任务、proxy/pretraining、鲁棒性预训练和人工检查四类，并给出每类动作进入 LLM 进化时应优先搜索的 reward、termination 和 sampling 杠杆。当前 ASAP 包没有显式后空翻、真实翻墙或钻洞文件名，因此这些目标在真实数据到位前只能使用相邻动作做预训练和算法压力测试，不能作为最终任务成功率结论。

如果只想检查闭环目录、prompt 和反馈接线，不启动训练：

```bash
python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep \
  --use_llm \
  --generations 2 \
  --population_size 2 \
  --skip_execute
```

## 运行时资源控制

候选训练失败不一定代表算法失败。`execute_generation.py` 会为每个候选写出：

```text
outputs/evolution*/<task>/<generation>/<genome_id>/status.json
```

如果状态为 `train_failed`、`eval_failed`、`train_exception` 或 `eval_exception`，`feedback_analyzer.py` 会把它并入下一代反馈。常见标签：

```text
runtime_sigbus
tensorboard_writer_failure
runtime_gpu_memory_pressure
runtime_disk_space
```

资源基因中已经加入：

```json
"resource": {
  "disable_logger": true
}
```

当该字段为 `true` 时，planner 会生成 `--disable_logger`，训练入口会使用 no-op summary writer。checkpoint、终端训练日志和最终评估不受影响，但不会启动 TensorBoard/W&B/Neptune 写线程。ASAP 进化配置默认使用该设置，以提高长队列稳定性。

正式脚本也支持：

```bash
DISABLE_LOGGER=1 bash scripts/run_asap_g1_evolution_experiments.sh
DISABLE_LOGGER=1 bash scripts/run_amass_g1_evolution_experiments.sh
```

## 动态资源晋级

V1.5 增加 stage2 晋级机制。`planner.py` 会为每个候选同时渲染 `train_stage1/eval_stage1` 与 `train_stage2/eval_stage2`。默认正式队列只执行 stage1；打开 `--enable_stage2` 后，执行器会先进行小预算训练和 16 回合评估，只有候选满足以下任一条件才继续投入 stage2 预算：

```text
stage1_success_rate >= evolution.stage1_success_threshold
success_delta_vs_baseline >= --stage2_min_success_delta
fitness_delta_vs_baseline >= --stage2_min_fitness_delta
```

未晋级候选会记录为 `evaluated_stage1_only`，不再消耗更大训练预算。同一 genome 同时存在 `eval_stage1.json` 与 `eval_stage2.json` 时，`scoreboard.py` 和 `feedback_analyzer.py` 优先使用 stage2 结果。

正式 ASAP 队列启用方式：

```bash
ENABLE_STAGE2=1 bash scripts/run_asap_g1_evolution_experiments.sh
```

扩展 ASAP 队列 `scripts/run_asap_g1_extended_experiments_after_default.sh` 默认启用 stage2，可在 default 队列结束后继续执行 `jump_forward_l4 / side_jump_l4 / CR7_dynamic` 等相邻高动态动作。

## LLM 输出抢救

MiniMax M3 在长 prompt 或多候选输出时可能出现后半段 JSON 截断。V1.5 的 `minimax_client.py` 增加了 complete-candidate salvage：如果顶层 JSON 无法解析，客户端会扫描 `candidates` 数组并保留已经闭合的完整 candidate 对象，再由本地 seed 补足剩余种群。

这样可以把“第二个候选截断”从整代失败降级为“部分候选可用”，对应任务书中对复杂算法生成可执行性差的约束修复。

已经产生 `llm_raw_text.txt` 的旧输出目录也可以离线恢复：

```bash
python scripts/evolution/recover_llm_candidates.py \
  --config evolution/configs/g1_asap_jump_forward_l5_v1.json \
  --raw_text outputs/evolution_asap/<task>/<generation>/llm_raw_text.txt \
  --output_dir outputs/evolution_asap_recovered/<task>/<generation> \
  --population_size 2 \
  --fill_with_local
```

## 算法级改动包

V1 默认让 Mimimax M3 只输出参数 genome。后续要让 LLM 参与“新增 reward/termination/采样策略”时，必须使用受限算法改动包：

```text
evolution/algorithm_patch_schema.json
```

允许的改动类型：

```text
reward_term
termination_term
sampling_policy
curriculum_rule
evaluation_metric
```

改动包只描述模板、输入、参数和安全检查，不允许 LLM 直接输出任意 Python 源码。进入训练前必须通过本地验证、`py_compile`、IsaacLab 1-iteration smoke test 和 16 episode 小评估。

验证改动包：

```bash
python scripts/evolution/patch_validator.py \
  --patch evolution/examples/crawl_ceiling_zone_reward_v1.json
```

任务特征统一描述模板：

```text
evolution/task_feature_schema.json
```

后续翻墙、钻洞、后空翻、登墙转身和新机器人迁移时，先写任务特征 profile，再由 LLM 根据任务特征和 baseline 反馈提出候选算法。

正式考核时，保留 baseline 与进化后最优候选各不少于 50 episode 的 motion-start 评估 JSON，并报告：

```text
success_rate_improvement = evolved_success_rate - baseline_success_rate
```

当前配置使用严格 128 episode baseline `0.9375`，目标是最终候选达到至少 `1.0175` 不可能，因此对于已接近饱和的 G1 50cm 膝爬，应把 V1 框架主要用于后续更难的翻墙/钻洞基线，并在该任务上比较自主进化前后的成功率提升。

## 特技动作扩展

已预注册三类后续动作任务：

```text
Tracking-Backflip-G1-v0
Tracking-WallTurn-G1-v0
Tracking-CrawlTunnel-G1-v0
```

对应配置：

```text
evolution/configs/g1_backflip_v1.json
evolution/configs/g1_wall_turn_v1.json
evolution/configs/g1_crawl_tunnel_v1.json
```

这些配置使用通用特技 prompt：

```text
evolution/prompts/mimimax_m3_stunt_candidate_generation_zh.md
```

新增可搜索 reward genes 包括：

```text
reward.apex_height_weight
reward.phase_progress_weight
reward.landing_stability_weight
reward.ceiling_clearance_weight
reward.yaw_alignment_weight
reward.contact_force_weight
```

特技 motion 数据源清单和落盘规范见：

```text
evolution/action_catalog/stunt_motion_sources_zh.md
```

ASAP G1 retargeted 数据已接入为任务套件：

```text
scripts/asap_g1_task_suite.py
scripts/create_asap_task_profiles.py
evolution/configs/g1_asap_jump_forward_l5_v1.json
evolution/configs/g1_asap_turn_jump_l5_v1.json
evolution/configs/g1_asap_spiderman_l2_v1.json
evolution/configs/g1_asap_single_foot_jump_l2_v1.json
```

默认 ASAP 正式队列：

```bash
cd /root/whole_body_tracking-main
TASK_IDS="$(python scripts/asap_g1_task_suite.py --list-default)" \
bash scripts/prepare_asap_g1_stunt_motions.sh

nohup bash scripts/run_asap_g1_evolution_experiments.sh \
  > logs/background/asap_g1_evolution_formal_$(date +%Y%m%d_%H%M).log 2>&1 &
```

队列完成后，finalizer 会为每个任务选择跨 generation 的最佳候选，执行 64 episode 复评、渲染视频，并生成 suite 级总表：

```bash
bash scripts/finalize_asap_evolution_results.sh
python scripts/evolution/summarize_asap_suite.py \
  --task_ids g1_asap_jump_forward_l5 g1_asap_turn_jump_l5 g1_asap_spiderman_l2 g1_asap_single_foot_jump_l2 \
  --include_interim
```

输出位置：

```text
artifacts/<task>/eval/evolution_summary_zh.md
artifacts/<task>/video/best_evolved_<genome_id>/rl-video-step-0.mp4
artifacts/asap_suite/evolution_suite_summary_zh.md
```

拿到后空翻 motion 后，可以先生成候选计划：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_backflip_v1.json \
  --use_llm \
  --dry_run \
  --population_size 4 \
  --generation 0 \
  --llm_timeout 300
```
