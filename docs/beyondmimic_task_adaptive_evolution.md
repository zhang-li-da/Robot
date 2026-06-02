# 面向障碍特技任务的 BeyondMimic 自主进化框架 V1.0

本文档对应任务书中的“任务驱动的特技动作学习算法自动设计与自主进化”算法部分功能 V1.0。当前仓库已经完成 Unitree G1 50cm 膝爬越障的 sim2sim 基线；下一阶段的目标不是泛化地改造 BeyondMimic，而是围绕 400m 障碍中的翻越矮墙、钻洞等任务特征，让模仿学习算法可以自动产生、验证、淘汰和改进任务适配版本。

## BeyondMimic 在本类任务上的主要缺陷

1. 单条高难度 motion clip 容易造成阶段覆盖不足。越障动作通常包含接近、接触、支撑、翻越、落地恢复等阶段，失败集中在某一阶段后，自适应采样会过度偏向局部失败片段，导致起始阶段和收尾阶段执行能力退化。

2. 原始模仿奖励没有显式表达任务成功。关节、刚体位姿和速度跟踪能够逼近参考动作，但不能直接保证“越过 50cm 障碍”“胸腹不撞墙”“钻洞时头背高度受控”等任务指标。

3. 终止条件对接触型动作过于敏感。原始 `ee_body_pos`、anchor 姿态和高度终止对普通全身跟踪合理，但对膝盖支撑、手膝接触、低姿态钻洞等任务，过早终止会让策略无法探索必要接触。

4. Reward 权重、PPO 参数、RSI/phase sampling、domain randomization 仍依赖人工炼丹。当前 50cm 膝爬实验成功依赖多轮人工调整 fixed-start、uniform phase、终止阈值和评估逻辑，说明算法本身缺少闭环诊断和自动调参。

5. 跨机器人抽象不足。G1、H1、其他人形机器人的关节数量、限位、末端命名、躯干比例、执行器能力不同；直接复用单一配置会把“运动模仿问题”混入“机器人形态适配问题”。

6. sim2real 鲁棒性还没有进入搜索目标。真实机器人需要限制扭矩峰值、动作频率、接触冲击、姿态恢复余量和传感延迟；如果训练只优化仿真成功率，后续 sim2real 风险较高。

7. 缺少失败诊断到算法修改的自动路径。已有评估 JSON 记录了成功率、最大前进距离、离地高度、终止原因，但这些信息尚未自动反馈到下一代 reward、采样和训练资源分配。

## V1 框架目标

V1 框架采用“受控 LLM + 进化策略”的方式：

1. 初始化：从当前 BeyondMimic 基线和已验证 G1 膝爬配置出发，生成一组结构化算法基因。

2. 交叉：在通过验证的候选之间组合 reward、采样、PPO、终止条件、domain randomization 和资源配置。

3. 变异：在安全范围内扰动关键基因，形成更贴合翻墙、钻洞、膝爬等任务的候选。

4. 动态评估：先用小预算筛掉明显失败候选，再把更多训练迭代和评估 episode 分配给高潜力候选。

5. 淘汰机制：用不少于 50 次任务执行的成功率、任务进度、终止原因和能耗代理指标计算 fitness，保留 top-k。

6. LLM 约束：Mimimax M3 只输出 JSON 格式的算法基因，不直接修改任意 Python 文件；本地 validator 负责 schema、范围、任务不变量和命令可执行性检查。

## 分层化算法表示

算法基因分为六层：

1. `reward`：模仿项权重与 std、动作平滑、关节限位、非期望接触、任务进度项占比。

2. `sampling`：adaptive sampling、uniform phase、fixed-start 概率、固定起始帧范围。

3. `termination`：anchor 高度、anchor 姿态、末端/关键刚体高度误差阈值。

4. `ppo`：学习率、熵系数、KL、clip、GAE、网络宽度和激活函数。

5. `domain_randomization`：摩擦、关节默认位姿扰动、躯干质心扰动、外力 push 区间。

6. `resource`：阶段训练迭代、并行环境数、评估 episode、晋级阈值和最终统计次数。

这种表示可以覆盖 BeyondMimic 当前主要人工调节点，同时避免 LLM 生成无法执行的大段代码。

## Fitness 设计

最终考核指标要求：相较于自主进化前的基准强化学习或模仿学习算法，单次任务执行成功率提升大于 8%，测试不少于 50 次。

V1 fitness 采用：

```text
fitness =
  100.0 * success_rate
  + 8.0 * clipped(mean_max_torso_x / target_x, 0, 1)
  + 4.0 * clipped(mean_max_clearance_over_obstacle / 0.20, -1, 1)
  + 2.0 * clipped(mean_return / 40.0, -1, 1)
  - 2.0 * failure_rate_from_anchor_pos
  - 1.5 * failure_rate_from_ee_body_pos
```

翻墙任务可替换 `target_x` 和 clearance 目标；钻洞任务需要额外加入 `max_head_height_below_ceiling`、`torso_pitch_window`、`no_ceiling_collision` 等指标。

## Mimimax M3 在闭环中的作用

Mimimax M3 负责提出候选算法基因和候选修改理由，但不直接执行训练，也不直接 patch 任意源码。完整链路为：

```text
历史实验结果 + 结构化失败反馈 + 当前任务描述 + 安全搜索空间
        -> Mimimax M3 JSON 候选
        -> 本地 schema/range/invariant 校验
        -> 渲染训练命令和评估命令
        -> 小预算训练/评估
        -> score board
        -> feedback analyzer
        -> 交叉/变异/淘汰
        -> 下一代候选
```

## V1.2 闭环反馈层

当前新增了两类运行时组件：

```text
scripts/evolution/feedback_analyzer.py
scripts/evolution/closed_loop.py
```

`feedback_analyzer.py` 把每个候选的 `eval_*.json` 转成结构化失败反馈：

```text
failure_tags
hypotheses
suggested_levers
dominant_termination
baseline_delta
recommendation
```

例如本次 `gen0_m3_000` 候选完成 2048 环境、800 iteration 训练后，motion-start 16 回合评估为 0/16 成功，所有 episode 因 `ee_body_pos` 终止，平均最大前进距离约 0.36m，平均越障 clearance 仍为负。该结果会被标记为：

```text
no_success
early_progress_failure
insufficient_clearance
ee_body_pos_dominant
deterministic_collapse
severe_regression_vs_baseline
```

这类反馈会进入下一代 prompt，要求 LLM 不再重复相同退化策略，而是围绕合法接触、前进阶段、phase sampling、终止阈值和 baseline-adjacent repair 进行修正。

`closed_loop.py` 是多代编排器，负责：

```text
run_generation.py -> execute_generation.py -> scoreboard.json -> feedback.json -> 下一代 run_generation.py
```

最小命令：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep \
  --use_llm \
  --generations 3 \
  --population_size 4 \
  --stop_on_target
```

只检查闭环，不启动训练：

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

每一代的状态记录在：

```text
outputs/evolution/closed_loop_<timestamp>/loop_state.json
```

## 受限算法改动包

为了从“参数进化”扩展到“算法结构进化”，V1.2 增加了受限 patch schema：

```text
evolution/algorithm_patch_schema.json
```

允许 LLM 提出五类改动：

```text
reward_term
termination_term
sampling_policy
curriculum_rule
evaluation_metric
```

约束原则：

1. LLM 只输出 patch 描述，不直接输出任意 Python 源码。
2. patch 必须绑定失败标签或任务特征。
3. patch 只能使用模板允许的输入和数学操作。
4. 进入正式训练前必须通过本地 schema 检查、`py_compile`、1 iteration IsaacLab smoke test 和 16 episode 小评估。
5. patch 必须能通过配置开关回滚。

本地验证命令：

```bash
python scripts/evolution/patch_validator.py \
  --patch evolution/examples/crawl_ceiling_zone_reward_v1.json
```

任务特征输入模板：

```text
evolution/task_feature_schema.json
```

后续新增翻墙、钻洞、后空翻、登墙转身或新机器人时，先写 task feature profile，再运行闭环。这样 LLM 的搜索目标从“泛化调参”变成“围绕任务几何、合法接触、机器人能力和评估协议进行定向算法进化”。

## V1 到 V2 的扩展方向

1. 把受限 patch schema 落成自动代码生成器：从 JSON patch 生成 reward/termination 模板代码，并自动注册到 IsaacLab config。

2. 增加形态适配层：把机器人描述、关节限位、末端集合、执行器能力转换成统一 robot capability profile。

3. 增加任务特异 reward 函数库：翻墙接触序列、洞口高度约束、落地稳定、手脚支撑合法性。

4. 增加 sim2real 约束：扭矩裕度、动作频率、低通滤波、延迟扰动、接触冲击上限和跌倒风险评分。

5. 增加第三方测试包：固定随机种子、固定评估 motion-start/env-reset 两套协议、至少 50 episode，并自动生成基线与进化后对比报告。

## V1.1 特技动作扩展

当前仓库已经把“后空翻、登墙转身、钻洞”从临时实验需求提升为可注册任务规格：

```text
Tracking-Backflip-G1-v0
Tracking-WallTurn-G1-v0
Tracking-CrawlTunnel-G1-v0
```

对应进化配置：

```text
evolution/configs/g1_backflip_v1.json
evolution/configs/g1_wall_turn_v1.json
evolution/configs/g1_crawl_tunnel_v1.json
```

新增 reward genes：

```text
apex_height_weight
phase_progress_weight
landing_stability_weight
ceiling_clearance_weight
yaw_alignment_weight
contact_force_weight
```

这些项解决 BeyondMimic 基线对高动态特技动作的三个缺陷：

1. 后空翻需要鼓励腾空高度与稳定落地，而不是只按每帧姿态误差奖励。
2. 登墙转身需要显式检查前进、越墙/触墙、最终朝向和落地恢复。
3. 钻洞需要把“低姿态通过且不撞顶”作为任务目标，而不是让 anchor 高度终止误杀动作。
4. 对 `crawl_progress_stall`、`mid_phase_progress_failure` 这类“没有摔倒但卡在动作中段”的失败，`phase_progress_weight` 提供了直接推进 reference motion 后段的可进化杠杆。

数据源候选清单见：

```text
evolution/action_catalog/stunt_motion_sources_zh.md
```

拿到新 motion 后的最小闭环：

```bash
cd /root/whole_body_tracking-main

python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_backflip_v1.json \
  --use_llm \
  --dry_run \
  --population_size 4 \
  --generation 0 \
  --llm_timeout 300

python scripts/evolution/execute_generation.py \
  --config evolution/configs/g1_backflip_v1.json \
  --output_dir outputs/evolution/<run_id> \
  --baseline_eval artifacts/g1_backflip/evaluation/baseline_eval.json \
  --baseline_id g1_backflip_baseline
```

如果 IsaacLab/Vulkan 图形设备不可用，执行器会写 `blocked_environment.json`，候选不会被误记为算法失败。

## V1.3 ASAP 动作库接入

用户提供的 `/root/ASAP-main.zip` 已解压到：

```text
/root/ASAP-main
```

`/root/shared-nvme/datasets/ASAP-main` 仅作为兼容回退路径，正式索引和任务画像默认使用 `/root/ASAP-main`。

其中可直接使用的 G1 retargeted 动作位于：

```text
/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles
```

ASAP 的 G1 动作是 23DoF，而当前 IsaacLab G1 是 29DoF。`scripts/rebuild_g1_motion_isaaclab.py` 已扩展支持 ASAP `.pkl`：

```text
root_trans_offset
root_rot  # ASAP 为 xyzw，转换时改成 IsaacLab/BeyondMimic 使用的 wxyz
dof       # 23DoF，缺失的 G1 wrist joints 使用默认角补齐
fps
```

动作目录由下面脚本自动生成：

```bash
cd /root/whole_body_tracking-main
conda activate /root/shared-nvme/conda_envs/isaaclab210
python scripts/index_asap_motion_catalog.py
python scripts/index_asap_assets.py
python scripts/select_asap_evolution_tasks.py --limit 24
python scripts/create_asap_task_profiles.py
python scripts/build_asap_task_adaptive_roadmap.py --limit 18
```

输出文件：

```text
evolution/action_catalog/asap_motion_catalog.json
evolution/action_catalog/asap_motion_catalog_zh.md
evolution/action_catalog/asap_asset_manifest.json
evolution/action_catalog/asap_asset_manifest_zh.md
evolution/action_catalog/asap_evolution_candidate_queue.json
evolution/action_catalog/asap_evolution_candidate_queue_zh.md
evolution/action_catalog/asap_task_adaptive_roadmap.json
evolution/action_catalog/asap_task_adaptive_roadmap_zh.md
evolution/task_profiles/g1_asap_*.json
```

当前目录统计为 52 个动作片段，其中包括 16 个跳跃/转体跳候选、10 个登墙/转向 proxy 候选、1 个 SpiderMan 低姿态大幅动作候选。注意：该 ASAP 包当前没有显式 `backflip`/后空翻文件；后空翻仍需要后续接入 CMU/SFU/DeepMimic/ASE 等真实后空翻 motion。ASAP 的单脚跳只能作为高动态起跳/落地 proxy，不能作为真实后空翻结果报告。

已转换并核验的代表动作：

```text
artifacts/g1_asap_jump_forward_l5/motion/motion.npz
  202 frames, 50Hz, 29DoF, x displacement ~= 1.97m, root z ~= 0.49-0.99m

artifacts/g1_asap_turn_jump_l5/motion/motion.npz
  222 frames, 50Hz, 29DoF, x displacement ~= 0.78m, root z ~= 0.74-1.28m

artifacts/g1_asap_spiderman_l2/motion/motion.npz
  262 frames, 50Hz, 29DoF, low-pose proxy, root z ~= 0.45-0.96m

artifacts/g1_asap_single_foot_jump_l2/motion/motion.npz
  282 frames, 50Hz, 29DoF, high-dynamic proxy, root z ~= 0.74-1.20m
```

ASAP 任务套件的单一来源文件为：

```text
scripts/asap_g1_task_suite.py
```

其中每条任务规格包含：

```text
source              # ASAP pkl 源文件
artifact            # 转换后的 motion.npz
isaac_task          # IsaacLab 注册任务
baseline_task       # BeyondMimic flat baseline 任务
convert_flags       # 位移对齐/初始朝向归零等转换选项
success_criteria    # 评估协议和 LLM 搜索目标
motion_catalog_filter_tasks
```

后续新增真实后空翻、翻墙、钻洞、登墙转身数据时，先在 `TASK_SPECS` 中增加一条规格，再运行配置生成和正式实验脚本。

对应配置入口：

```text
evolution/configs/g1_asap_jump_forward_l5_v1.json
evolution/configs/g1_asap_jump_forward_l4_v1.json
evolution/configs/g1_asap_side_jump_l4_v1.json
evolution/configs/g1_asap_turn_jump_l5_v1.json
evolution/configs/g1_asap_spiderman_l2_v1.json
evolution/configs/g1_asap_single_foot_jump_l2_v1.json
evolution/configs/g1_asap_cr7_l2_dynamic_v1.json
```

这些配置通过 `task.motion_catalog` 把动作目录注入 Mimimax M3 prompt，并通过 `task.motion_catalog_filter_tasks` 只保留相关动作统计。LLM 因此能看到每类动作的水平位移、root 高度范围、标签和 proxy 边界，再决定 reward、sampling、termination 和 PPO 候选，而不是只根据任务名称泛化调参。

`scripts/create_asap_task_profiles.py` 会把每个 ASAP 任务进一步转成任务特征 profile，字段包括：

```text
task_identity
robot_profile
motion_profile
environment_profile
success_criteria
legal_contacts
risk_controls
baseline_contract
```

这些 profile 是后续让 LLM 从“任务特征 + 基线缺陷 + 失败反馈”生成算法改动包的稳定输入。

`scripts/create_asap_evolution_configs.py` 会根据 `reward_terms` 自动打开对应搜索空间。例如 `g1_asap_turn_jump_l5` 会开放：

```text
reward.apex_height_weight
reward.phase_progress_weight
reward.landing_stability_weight
reward.yaw_alignment_weight
reward.contact_force_weight
```

避免从 `wall_turn` 基配置继承时把 `apex_height_weight` 锁死为 `[0, 0]`，导致 LLM 明知道任务需要腾空高度却无法生成可执行覆盖参数。

代表性 dry-run 可以使用单代生成器，也可以使用完整闭环编排器。
推荐先用闭环编排器验证“任务配置 -> Mimimax M3 -> genome -> train/eval command”的全链路，但不执行训练：

```bash
cd /root/whole_body_tracking-main
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210
export PYTHONPATH=/root/whole_body_tracking-main/source/whole_body_tracking:${PYTHONPATH:-}

python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_asap_turn_jump_l5_v1.json \
  --output_root outputs/evolution_asap_dryrun/turn_jump_l5_llm_phase \
  --generations 1 \
  --population_size 1 \
  --use_llm \
  --llm_timeout 600 \
  --skip_execute
```

也可以只检查 `run_generation.py`：

```bash
python scripts/evolution/run_generation.py \
  --config evolution/configs/g1_asap_jump_forward_l5_v1.json \
  --output_root outputs/evolution_asap_dryrun/jump_forward_l5 \
  --dry_run \
  --population_size 1 \
  --generation 0
```

生成的输出目录中会包含：

```text
motion_catalog_snapshot.json
config_snapshot.json
plans/*.json
train_commands.sh
eval_commands.sh
```

后续正式实验可以从 ASAP proxy baseline 开始，再进入 LLM 辅助闭环：

```bash
cd /root/whole_body_tracking-main
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210

# 默认队列：forward jump L5、turn jump L5、SpiderMan proxy、single-foot jump proxy。
nohup bash scripts/run_asap_g1_evolution_experiments.sh \
  > logs/background/asap_g1_evolution_formal_$(date +%Y%m%d_%H%M).log 2>&1 &
```

如果只运行某几个任务：

```bash
TASK_IDS="g1_asap_jump_forward_l5 g1_asap_turn_jump_l5" \
EVO_GENERATIONS=2 EVO_POPULATION=2 NUM_ENVS=2048 \
bash scripts/run_asap_g1_evolution_experiments.sh
```

该脚本会按顺序执行：

```text
index_asap_motion_catalog.py
create_asap_evolution_configs.py
create_asap_task_profiles.py
prepare_asap_g1_stunt_motions.sh
train flat BeyondMimic baseline
train task-adapted baseline
eval_stunt.py baseline/adapted
closed_loop.py Mimimax M3 evolution
```

在单张 RTX 3090 上，建议等当前 AMASS 正式实验完成后再启动 ASAP 正式队列；否则多个 IsaacLab 进程会争用 GPU 和 Vulkan/PhysX 资源。

如果已经有 AMASS 训练占用 GPU，可以把 ASAP 正式队列挂成等待任务：

```bash
cd /root/whole_body_tracking-main
mkdir -p logs/background
nohup bash -lc '
cd /root/whole_body_tracking-main
while pgrep -f "[r]un_amass_g1_evolution_experiments.sh|[c]losed_loop.py --config evolution/configs/g1_jump_leap_v1.json|[t]rain.py --task Tracking-JumpLeap-G1-v0" >/dev/null; do
  date
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  sleep 300
done
date
env NUM_ENVS=2048 BASELINE_ITERS=800 ADAPTED_ITERS=800 EVO_GENERATIONS=2 EVO_POPULATION=2 LOGGER=tensorboard bash scripts/run_asap_g1_evolution_experiments.sh
' > logs/background/asap_wait_then_formal_$(date +%Y%m%d_%H%M).log 2>&1 &
```

LLM 客户端日志只允许记录 URL、API mode 和 model，不记录 API key 或 key 前后缀；如果生成日志中出现旧格式 `key=...`，提交仓库前必须先替换为 `key=<redacted>`。

## V1.6 ASAP 任务族路线图

新增 ASAP 动作数据后，框架不直接把所有 clip 都当成正式任务训练，而是先通过：

```text
motion catalog -> candidate queue -> task adaptive roadmap -> LLM prompt context
```

把动作分为四类：

```text
formal_or_curriculum
  可作为正式实验或相邻课程任务，例如 forward jump、turn jump。

proxy_pretraining
  只能作为预训练或压力测试，例如 SpiderMan wall-contact proxy、single-foot jump flip proxy、squat low-posture proxy。

robustness_pretraining
  用于改进落地恢复、单腿支撑、全身协调和 domain randomization，不作为目标任务成功证据。

manual_gate
  动作语义不明确，需要人工确认后再写 task profile。
```

路线图生成入口：

```bash
python scripts/build_asap_task_adaptive_roadmap.py --limit 18
```

输出：

```text
evolution/action_catalog/asap_task_adaptive_roadmap.json
evolution/action_catalog/asap_task_adaptive_roadmap_zh.md
```

`asap_task_adaptive_roadmap_zh.md` 会给每个任务族写出 LLM 应优先搜索的算法杠杆。例如：

```text
aerial_turn_jump
  reward: task_progress, phase_progress, apex_height, yaw_alignment, landing_stability
  sampling: 保留起跳、空中、落地阶段覆盖
  termination: 训练阶段放宽空中姿态/ee 容忍，最终 yaw 标准不能放宽

low_posture_pretraining
  reward: phase_progress, ceiling_clearance, landing_stability
  sampling: 覆盖低姿态进入、保持、退出阶段
  termination: 不能因为 root height 低就提前终止

recovery_pretraining
  reward: phase_progress, landing_stability, contact_force
  sampling: 给跳跃、转体、翻墙后的失败落地阶段补充采样
  termination: 保留最终速度和角速度门槛
```

这一步的作用是把 LLM 从“泛化调参器”约束成“任务族诊断器”：它必须根据动作标签、位移、高度变化、合法接触和历史失败标签选择 reward、termination、sampling、PPO 或资源基因，而不是只输出通用 BeyondMimic 参数组合。

当前 ASAP 包没有真实后空翻、真实翻墙和真实钻洞文件名，因此 roadmap 会显式保留数据缺口。后续用户提供 400m 障碍翻墙 mesh/钻洞 mesh、真实翻墙/钻洞跟踪数据、新机器人 URDF/MJCF 后，应先刷新 catalog 和 roadmap，再启动正式闭环；proxy 动作产生的成功率不能用于最终任务书考核。

## 运行时失败反馈与资源基因

ASAP 和 AMASS 的正式训练中，候选体可能不是因为算法性能差而失败，而是因为运行时资源异常中断，例如 `SIGBUS`、TensorBoard event writer 线程异常、GPU 显存压力或磁盘写入失败。V1.4 已把这类失败纳入闭环：

```text
execute_generation.py
  每个候选写出 status.json，记录 train_failed/eval_failed、signal、return_code、log tail。

feedback_analyzer.py
  读取 eval_*.json 的同时扫描 status.json。
  将 runtime_sigbus、tensorboard_writer_failure、runtime_gpu_memory_pressure 等标签写入 feedback.json。

Mimimax prompt
  如果 llm_feedback_brief.runtime_failures 非空，下一代必须先生成运行时修复候选。
```

新增资源基因：

```json
"resource": {
  "disable_logger": true
}
```

`resource.disable_logger=true` 时，`planner.py` 会给训练命令加入 `--disable_logger`。`scripts/rsl_rl/train.py` 会使用 no-op summary writer，保留终端日志、配置 dump 和 checkpoint，但不启动 TensorBoard/W&B/Neptune 写线程。这样可以避免把日志器或 IO 异常误判成 reward/termination 设计失败。

ASAP 进化配置默认强制：

```json
"resource_defaults": {
  "disable_logger": true
}
```

因此正式对比流程为：

```text
baseline/adapted task-reward 训练：可继续使用 tensorboard，方便观察曲线。
LLM evolution candidates：默认 disable_logger，优先保证大规模队列稳定和 checkpoint 可复评。
final 64 episode eval：不受 logger 设置影响，仍按同一成功标准统计。
```

## V1.4 闭环续跑与中段停滞修复

`closed_loop.py` 支持从已有一代结果继续：

```bash
python scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_crawl_tunnel_v1.json \
  --output_root outputs/evolution_amass/crawl_tunnel_continue \
  --baseline_eval artifacts/g1_crawl_tunnel/eval/baseline_beyondmimic.json \
  --baseline_id baseline_beyondmimic \
  --start_generation 2 \
  --generations 2 \
  --population_size 2 \
  --initial_history outputs/evolution_amass/crawl_tunnel/20260602_153156_521231_gen01/scoreboard.json \
  --initial_feedback outputs/evolution_amass/crawl_tunnel/20260602_153156_521231_gen01/feedback.json \
  --use_llm \
  --llm_timeout 600
```

这个入口用于继续利用上一代的失败标签和最好候选，不需要从 gen0 重新开始。针对当前 crawl_tunnel 结果，已修正两点：

```text
scoreboard.py
  crawl 未进入 ceiling zone 时不再获得虚假的 clearance 加分

RewardGenes / planner / G1*RewardsCfg
  新增 phase_progress_weight，用于处理 mid-phase/crawl progress stall
```

## V1.5 最终复评与考核摘要

小预算 `stage1` 评估只用于候选筛选，不能直接作为任务书考核结果。正式报告必须额外运行不少于 50 回合的最终复评，并把结果传入 `summarize_task_evolution.py`：

```bash
python -u scripts/evolution/summarize_task_evolution.py \
  --config evolution/configs/g1_jump_leap_v1.json \
  --evolution_root outputs/evolution_amass/jump_leap \
  --baseline_eval artifacts/g1_jump_leap/eval/baseline_beyondmimic.json \
  --adapted_eval artifacts/g1_jump_leap/eval/adapted_task_rewards.json \
  --final_eval artifacts/g1_jump_leap/eval/best_evolved_64ep.json \
  --final_label gen0_m3_001_final64 \
  --output_json artifacts/g1_jump_leap/eval/evolution_summary.json \
  --output_md artifacts/g1_jump_leap/eval/evolution_summary_zh.md
```

摘要中的 `final_target_check` 字段用于直接判断任务书指标：

```text
minimum_trials_met
target_improvement_met
target_met
success_rate_delta_vs_baseline
```

当前 `g1_jump_leap` 的 64 回合最终复评为 `35/64`，baseline 为 `0/64`，成功率提升 `0.546875`，已满足“提升 >8%，测试不少于 50 次”的阶段指标。
