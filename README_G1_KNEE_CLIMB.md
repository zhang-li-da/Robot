# Unitree G1 50cm 膝爬越障项目说明

这是基于 BeyondMimic / whole-body tracking 框架扩展的 Unitree G1 50cm 膝爬越障模仿学习实验仓库。

原始框架来自 BeyondMimic motion tracking code。本仓库在其基础上加入：

- Unitree G1 50cm 膝爬越障任务 `Tracking-KneeClimb-G1-v0`
- 50cm 障碍物仿真环境
- G1 动作重定向后的参考轨迹
- 固定起点 motion-start 评估脚本
- 成功视频渲染脚本
- 最终 checkpoint、评估 JSON、视频和实验报告

## 快速查看结果

最终报告：

- `reports/g1_knee_climb_50cm_report_zh.pdf`
- `reports/g1_knee_climb_50cm_report_zh.md`

最终 checkpoint：

```text
artifacts/g1_knee_climb_50cm/checkpoints/model_6000.pt
```

成功视频：

```text
artifacts/g1_knee_climb_50cm/video/rl-video-step-0.mp4
```

评估 JSON：

```text
artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_64ep.json
artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json
```

## 环境摘要

已验证环境：

- Ubuntu/Linux x86_64
- NVIDIA RTX 3090 24GB
- NVIDIA Driver 550.54.14
- Python 3.10.20
- Isaac Sim 4.5.0
- Isaac Lab 2.1.0
- PyTorch 2.5.1
- rsl-rl-lib 2.2.4

详细配置见：

```text
reports/g1_knee_climb_50cm_report_zh.pdf
```

## Git 管理策略

仓库只提交：

- 代码和配置；
- 训练、评估、渲染脚本；
- 报告；
- 精选最终 artifact。

仓库不提交：

- 完整 `logs/`；
- `outputs/`；
- TensorBoard event 文件；
- 原始下载 zip/tar 包；
- Isaac Lab 安装目录；
- conda 环境；
- 大型 robot description assets。

二进制 artifact 使用 Git LFS 管理，包括：

- `*.pt`
- `*.mp4`
- `*.npz`
- `*.pdf`

## 继续迭代建议

每次新实验建议新建分支：

```bash
git checkout -b exp/fixed-start-curriculum-v2
```

实验结束后只复制精选结果到：

```text
artifacts/g1_knee_climb_50cm/
```

然后更新：

```text
EXPERIMENTS.md
reports/
```

再提交代码和结果摘要。

