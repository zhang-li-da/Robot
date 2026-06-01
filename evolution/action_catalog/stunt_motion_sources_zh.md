# 特技动作模仿数据源候选清单

本文档记录后续后空翻、登墙转身、翻越矮墙、钻洞等动作可优先检索和下载的公开动作数据源。进入训练前，所有数据都需要统一转换为仓库的 `motion.npz` 格式，并通过 G1/H1/目标机器人关节限位检查。

## 优先数据源

| 数据源 | 适用动作 | 文件形态 | 用法 |
| --- | --- | --- | --- |
| CMU Graphics Lab Motion Capture Database | 后空翻、侧空翻、翻滚、体操、breakdance、跳跃落地 | ASF/AMC、BVH 转换版本 | 优先检索 `flip`、`cartwheel`、`roll`、`jump`、`acrobatics` 标签，作为高动态空中特技 reference |
| SFU Motion Capture Database | parkour roll、speed vault、monkey vault、障碍跳跃、bench running | BVH/FBX/C3D | 优先用于翻越矮墙、跨越障碍、落地恢复和手脚支撑动作 |
| AMASS + BABEL | 带语义标签的人体动作检索层 | SMPL/SMPL-H NPZ + action labels | 用 BABEL 标签检索 `backflip`、`cartwheel`、`climb`、`crawl`、`jump`，再回到 AMASS motion 做 retarget |
| KIT Whole-Body Human Motion Database | 带物体/环境交互的全身动作 | MMM/XML、C3D 等 | 用于有明确物体关系的钻洞、爬越、支撑动作，适合补足接触阶段 |
| DeepMimic / AMP / ASE 示例动作 | 后空翻、跳跃、踢腿等标准 humanoid benchmark | JSON、NPZ 或项目自定义格式 | 用于 smoke test：验证 retarget、reward、评估脚本能否完成高动态动作闭环 |

源站入口：

- CMU: https://mocap.cs.cmu.edu/
- SFU: https://mocap.cs.sfu.ca/
- BABEL: https://babel.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/
- KIT: https://motion-database.humanoids.kit.edu/

已确认 SFU 站点直接提供 BVH/FBX/C3D，且在 `Interaction with Obstacles` 下包含 `JumpAndRoll`、`HopOverObstacle`、`JumpOverObstacle`、`SpeedVault`、`MonkeyVault`、`RunningOnBench` 等条目，适合优先做翻越矮墙和登墙转身的原始 motion 候选。

## 下载后落盘规范

```text
raw_motions/
  cmu/
    README_SOURCE.md
    backflip/
    cartwheel/
  sfu/
    README_SOURCE.md
    parkour_vault/
    monkey_vault/
  amass_babel/
    README_SOURCE.md
    backflip/
    crawl/
  kit/
    README_SOURCE.md
    crawl_under/
artifacts/
  g1_backflip/
    motion/motion.npz
    config/retarget.yaml
    evaluation/
  g1_wall_turn/
    motion/motion.npz
    config/retarget.yaml
    evaluation/
  g1_crawl_tunnel/
    motion/motion.npz
    config/retarget.yaml
    evaluation/
```

`README_SOURCE.md` 必须记录下载 URL、license/使用限制、原始动作编号、动作帧率、是否包含手/足/膝接触信息。不要把不允许再分发的原始大文件提交到 Git。

## 检索关键词

后空翻：

```text
CMU mocap back flip BVH
CMU mocap gymnastics flips cartwheel
AMASS BABEL backflip cartwheel jump
DeepMimic backflip motion data
ASE humanoid backflip motion
```

登墙转身/翻越矮墙：

```text
SFU mocap parkour vault BVH
parkour monkey vault motion capture
speed vault motion capture BVH
AMASS BABEL climb vault obstacle
KIT whole-body climb over obstacle
```

钻洞/低姿态通过：

```text
AMASS BABEL crawl under
KIT whole-body crawl under table
motion capture crawl under obstacle
CMU mocap crawl BVH
```

## 纳入进化实验的准入条件

1. 动作转换后至少能在 `scripts/replay_npz.py` 中离线重放并通过关节限位检查。
2. motion clip 必须包含完整动作阶段：准备、主动作、恢复；如果缺少恢复阶段，需要人工裁剪或拼接静止恢复段。
3. 对接触型动作必须标记允许接触的身体部位，例如后空翻只允许脚落地，钻洞允许手/膝接触，登墙转身允许脚/手与墙接触。
4. 每个新动作必须建立 baseline 评估 JSON，后续自主进化提升按不少于 50 次任务执行计算。
