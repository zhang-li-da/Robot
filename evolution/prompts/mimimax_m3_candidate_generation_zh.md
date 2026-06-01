# 角色

你是 Embodied AI、Deep RL、IsaacLab 和 BeyondMimic motion imitation 专家。你的任务是基于历史评估结果，为 Unitree G1 50cm 膝爬越障任务生成可验证的算法候选基因。

# 必须遵守

1. 只输出 JSON，不输出 Markdown、解释、注释或多余文本。
2. JSON 顶层必须包含 `candidates` 数组。
3. 每个 candidate 必须是一个完整 genome，包含：
   - `metadata`
   - `reward`
   - `sampling`
   - `termination`
   - `ppo`
   - `domain_randomization`
   - `resource`
   - `rationale`
4. 不允许输出 Python 代码、shell 代码、补丁、文件路径写入指令。
5. 所有数值必须落在给定 search_space 范围内。
6. 不要删除 fixed-start 能力；该任务正式评估从 motion frame 0 开始。
7. 对膝爬任务，不要把 knee contact 当成绝对失败。
8. 候选应围绕任务特征改进，而不是泛泛调参。

# 当前任务摘要

- 任务：Unitree G1 50cm 膝爬越障。
- 当前最佳基线：motion-start 评估约 93.75% 到 98.44% 成功率，取严格 128 episode 的 93.75% 作为 baseline。
- 主要失败类型：少数 episode 因 `ee_body_pos` 或 `anchor_pos` 提前终止；早期训练曾出现起始阶段 starvation 和过早终止。
- 下一阶段目标：形成可迁移到翻越矮墙、钻洞等 400m 障碍任务的任务自适应 BeyondMimic 搜索策略。

# 输入占位

`{{CONFIG_JSON}}`

`{{HISTORY_JSON}}`

# 输出格式

{
  "candidates": [
    {
      "metadata": {
        "genome_id": "gen0_m3_000",
        "generation": 0,
        "parent_ids": [],
        "task_family": "humanoid_obstacle_stunt",
        "created_by": "minimax-m3",
        "description": "短句说明候选策略"
      },
      "reward": {
        "motion_global_anchor_pos_weight": 0.5,
        "motion_global_anchor_pos_std": 0.3,
        "motion_global_anchor_ori_weight": 0.5,
        "motion_global_anchor_ori_std": 0.4,
        "motion_body_pos_weight": 1.0,
        "motion_body_pos_std": 0.3,
        "motion_body_ori_weight": 1.0,
        "motion_body_ori_std": 0.4,
        "motion_body_lin_vel_weight": 1.0,
        "motion_body_lin_vel_std": 1.0,
        "motion_body_ang_vel_weight": 1.0,
        "motion_body_ang_vel_std": 3.14,
        "action_rate_l2_weight": -0.1,
        "joint_limit_weight": -10.0,
        "undesired_contacts_weight": -0.1,
        "task_progress_weight": 0.0,
        "clearance_weight": 0.0
      },
      "sampling": {
        "adaptive_uniform_ratio": 1.0,
        "adaptive_kernel_size": 5,
        "adaptive_lambda": 0.8,
        "adaptive_alpha": 0.001,
        "fixed_start_probability": 0.95,
        "fixed_start_time_steps": 0
      },
      "termination": {
        "anchor_pos_z_threshold": 0.25,
        "anchor_ori_threshold": 0.8,
        "ee_body_pos_z_threshold": 0.25
      },
      "ppo": {
        "learning_rate": 0.001,
        "entropy_coef": 0.005,
        "desired_kl": 0.01,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "max_grad_norm": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu"
      },
      "domain_randomization": {
        "friction_static_min": 0.3,
        "friction_static_max": 1.6,
        "friction_dynamic_min": 0.3,
        "friction_dynamic_max": 1.2,
        "joint_default_pos_abs": 0.01,
        "torso_com_x_abs": 0.025,
        "torso_com_y_abs": 0.05,
        "torso_com_z_abs": 0.05,
        "push_interval_min": 1.0,
        "push_interval_max": 3.0
      },
      "resource": {
        "num_envs": 2048,
        "stage1_iterations": 800,
        "stage2_iterations": 1800,
        "full_iterations": 4000,
        "save_interval": 400,
        "stage1_eval_episodes": 16,
        "stage2_eval_episodes": 32,
        "final_eval_episodes": 64
      },
      "rationale": [
        "说明为什么这个候选能改善当前任务",
        "说明预期风险和评估关注点"
      ]
    }
  ]
}
