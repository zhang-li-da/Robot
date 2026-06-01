# BeyondMimic 任务自适应算法自动进化框架

该目录保存“LLM + 进化策略”的 V1 框架，用于围绕 G1 膝爬、400m 障碍翻墙、钻洞等任务自动搜索 BeyondMimic 的任务适配参数组合。

## 设计原则

- Mimimax M3 只生成结构化 JSON 算法基因，不直接修改任意 Python 源码。
- 本地 validator 检查 schema、数值范围、任务不变量和安全边界。
- planner 将合法 genome 渲染成训练命令、评估命令和资源计划。
- scoreboard 读取评估 JSON，计算 fitness 并记录淘汰依据。
- API key 只从环境变量或 `/root/Mimimax` 读取，绝不提交到 Git。

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
  run_generation.py
docs/beyondmimic_task_adaptive_evolution.md
```

## 本地密钥配置

推荐使用环境变量：

```bash
export MINIMAX_API_URL="https://api.minimax.io/v1/text/chatcompletion_v2"
export MINIMAX_API_KEY="你的 key"
export MINIMAX_MODEL="MiniMax-M3"
```

如果不设置环境变量，脚本会尝试读取 `/root/Mimimax`。脚本只打印 key 是否存在和尾部短指纹，不打印完整 key。

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
  --llm_timeout 180
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
  --llm_timeout 180
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

正式考核时，保留 baseline 与进化后最优候选各不少于 50 episode 的 motion-start 评估 JSON，并报告：

```text
success_rate_improvement = evolved_success_rate - baseline_success_rate
```

当前配置使用严格 128 episode baseline `0.9375`，目标是最终候选达到至少 `1.0175` 不可能，因此对于已接近饱和的 G1 50cm 膝爬，应把 V1 框架主要用于后续更难的翻墙/钻洞基线，并在该任务上比较自主进化前后的成功率提升。
