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
历史实验结果 + 当前任务描述 + 安全搜索空间
        -> Mimimax M3 JSON 候选
        -> 本地 schema/range/invariant 校验
        -> 渲染训练命令和评估命令
        -> 小预算训练/评估
        -> score board
        -> 交叉/变异/淘汰
        -> 下一代候选
```

## V1 到 V2 的扩展方向

1. 增加任务特异 reward 函数库：翻墙接触序列、洞口高度约束、落地稳定、手脚支撑合法性。

2. 增加形态适配层：把机器人描述、关节限位、末端集合、执行器能力转换成统一 robot capability profile。

3. 增加 patch bundle 模式：只允许 LLM 在受限模板内新增 reward term 或 termination term，并通过单元测试和 IsaacLab smoke test 后才进入训练。

4. 增加 sim2real 约束：扭矩裕度、动作频率、低通滤波、延迟扰动、接触冲击上限和跌倒风险评分。

5. 增加第三方测试包：固定随机种子、固定评估 motion-start/env-reset 两套协议、至少 50 episode，并自动生成基线与进化后对比报告。
