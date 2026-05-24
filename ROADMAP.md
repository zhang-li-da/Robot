# 项目路线图：面向 400 米障碍的人形机器人特技动作学习

本仓库当前已经完成 Unitree G1 50cm 膝爬越障的 BeyondMimic 风格模仿学习实验。后续项目目标将从单一 G1 仿真任务扩展到多机器人、多障碍、多动作，以及“大模型 + 进化策略”的算法自主优化框架。

## 1. 长期目标

面向人形机器人完成 400 米障碍项目中的典型特技动作，重点包括：

- 翻越矮墙；
- 钻过洞口；
- 在不同人形机器人平台之间迁移，不局限于 Unitree G1；
- 先在仿真中实现稳定策略，再逐步推进真实机器人部署。

后续将输入：

- 翻墙任务 mesh / collision 数据；
- 钻洞任务 mesh / collision 数据；
- 翻墙与钻洞动作跟踪数据；
- 新机器人 URDF / MJCF / USD 资产；
- 新机器人关节限制、执行器参数、惯量和控制接口。

## 2. 算法研究目标

在 BeyondMimic 框架基础上，引入“大语言模型 + 进化策略”的自动设计与自主进化机制，将传统依赖专家经验的模仿学习和强化学习调参过程转化为自动化闭环。

核心研究方向：

- 神经网络结构搜索；
- 模仿学习参数优化；
- reward / termination / curriculum 自动改写；
- reference-state initialization 与 adaptive sampling 策略进化；
- 动作跟踪任务的自动诊断和代码生成；
- 基于评估结果的自动淘汰、交叉、变异和重训。

## 3. 任务书摘要

研究“大模型 + 进化策略”的迭代优化框架，将神经网络结构搜索、模仿学习参数优化等传统依赖专家经验的“炼丹”过程转化为自动化闭环。

框架基于“初始化 - 交叉 - 变异”的进化机制，通过研究：

- 分层化算法表示体系；
- 提示词工程；
- 动态评估资源调整；
- 淘汰机制；
- 复杂算法代码生成的可执行性检查；
- 需求响应覆盖率检查；
- 逻辑漏洞检测；

解决传统大语言模型在复杂算法代码生成中存在的逻辑漏洞、可执行性差、需求响应覆盖率低等问题，实现算法性能的自主迭代提升。

## 4. 阶段目标 V1.0

阶段目标：

- 初步实现“任务驱动的特技动作学习算法自动设计与自主进化”算法部分功能；
- 形成算法 V1.0；
- 在具体任务上验证自主进化前后成功率提升。

考核指标：

- 相较自主进化前的基准强化学习或模仿学习算法，单次任务执行成功率提升大于 8%；
- 测试不少于 50 次任务完成情况；
- 记录迭代优化前后任务成功率提升比例；
- 支持专家测试或第三方测试。

## 5. 当前基线

当前仓库中的基线任务：

```text
Unitree G1 50cm 膝爬越障
```

当前方法：

```text
BeyondMimic 风格 whole-body motion tracking + RSL-RL PPO
```

当前最佳 checkpoint：

```text
artifacts/g1_knee_climb_50cm/checkpoints/model_6000.pt
```

当前评估：

- 64 回合固定起点：63/64，成功率 98.44%
- 128 回合固定起点：120/128，成功率 93.75%

该结果可作为后续“大模型 + 进化策略”框架的初始 baseline 之一。

## 6. 推荐仓库结构

当前保留结构：

```text
artifacts/
  g1_knee_climb_50cm/
    checkpoints/
    config/
    evaluation/
    motion/
    video/

reports/
scripts/
source/
EXPERIMENTS.md
README_G1_KNEE_CLIMB.md
ROADMAP.md
```

后续建议扩展为：

```text
tasks/
  wall_climb/
    meshes/
    motions/
    configs/
    reports/
  tunnel_crawl/
    meshes/
    motions/
    configs/
    reports/

robots/
  unitree_g1/
  unitree_h1/
  other_humanoids/

evolution/
  prompts/
  candidates/
  evaluators/
  mutation_ops/
  selection/
  archives/

artifacts/
  <task_name>/
    checkpoints/
    evaluation/
    video/
    config/
```

说明：

- `source/` 保留可运行代码；
- `scripts/` 保留训练、评估、渲染、转换工具；
- `artifacts/` 只保留精选结果；
- 大型原始数据和全部历史 logs 不进入 Git；
- 大型二进制用 Git LFS 或外部对象存储管理；
- 每次正式实验必须在 `EXPERIMENTS.md` 中登记。

## 7. 未来迭代流程建议

每个新任务建议按以下流程推进：

1. 新建任务分支：

```bash
git checkout -b task/wall-climb-v1
```

2. 放入新任务资产：

- mesh / collision 数据；
- retarget 后 motion；
- robot asset；
- environment config。

3. 建立 baseline：

- 固定 BeyondMimic / PPO 配置；
- 至少 50 次评估；
- 保存 baseline 成功率。

4. 启动自主进化：

- LLM 生成 candidate 配置或代码补丁；
- 自动运行语法检查和小规模 smoke test；
- 通过后进入训练；
- milestone checkpoint 自动评估；
- 根据成功率、失败模式和资源消耗淘汰候选；
- 交叉 / 变异优胜候选；
- 记录完整 lineage。

5. 验证 V1.0 指标：

- 与 baseline 同任务、同评估标准对比；
- 至少 50 次测试；
- 成功率提升大于 8%；
- 输出报告和最终视频。

## 8. 版本管理规则

建议分支命名：

- `main`：稳定结果；
- `task/<task-name>`：任务开发；
- `exp/<experiment-name>`：具体实验；
- `evo/<task-name>-generation-<n>`：进化算法候选代；
- `robot/<robot-name>`：新机器人适配。

建议 tag：

- `g1-kneeclimb-v1.0`
- `wall-climb-baseline-v0`
- `wall-climb-evo-v1.0`
- `tunnel-crawl-baseline-v0`
- `tunnel-crawl-evo-v1.0`

每个正式 tag 应包含：

- checkpoint；
- motion；
- evaluation JSON；
- video；
- report；
- exact commit hash；
- environment notes。

## 9. 当前注意事项

- 本仓库当前没有提交完整 Isaac Lab 安装目录和 conda 环境；
- 新服务器需要按报告重新安装 Isaac Lab 2.1.0；
- `source/whole_body_tracking/whole_body_tracking/assets/unitree_description/` 被 `.gitignore` 排除，需要通过 `unitree_description.tar.gz` 或官方资源恢复；
- 当前 G1 50cm 膝爬任务是第一个可运行 baseline，不应被后续重构破坏；
- 后续新增机器人或任务时，优先保证已有 `Tracking-KneeClimb-G1-v0` 可继续评估。

