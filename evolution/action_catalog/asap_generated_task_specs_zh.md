# ASAP 自动生成任务 Spec

该文件把候选队列中未手写配置的高优先级动作转成可训练任务定义。

- 生成时间：`2026-06-03T01:25:56.490485+00:00`
- 最大生成数：`8`

| 任务ID | 源动作 | Isaac 任务 | base config | 成功类型 |
| --- | --- | --- | --- | --- |
| `g1_asap_turn_jump_l3` | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | `Tracking-WallTurn-G1-v0` | `g1_wall_turn_v1.json` | `progress` |
| `g1_asap_turn_jump_l2` | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | `Tracking-WallTurn-G1-v0` | `g1_wall_turn_v1.json` | `progress` |
| `g1_asap_squat_l2_lowposture` | `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass` | `Tracking-CrawlTunnel-G1-v0` | `g1_crawl_tunnel_v1.json` | `low_posture` |
| `g1_asap_turn_jump_l1` | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | `Tracking-WallTurn-G1-v0` | `g1_wall_turn_v1.json` | `progress` |
| `g1_asap_single_foot_jump_l1` | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | `Tracking-Backflip-G1-v0` | `g1_backflip_v1.json` | `backflip` |
| `g1_asap_jump_forward_l3` | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | `Tracking-JumpLeap-G1-v0` | `g1_jump_leap_v1.json` | `progress` |
| `g1_asap_squat_l1_lowposture` | `0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass` | `Tracking-CrawlTunnel-G1-v0` | `g1_crawl_tunnel_v1.json` | `low_posture` |
| `g1_asap_jump_forward_l2` | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | `Tracking-JumpLeap-G1-v0` | `g1_jump_leap_v1.json` | `progress` |

## 使用规则

- 自动生成任务默认不进入 `--list-default` 正式队列，需要显式传入 `TASK_IDS`。
- 带 `proxy_note` 的任务只能用于预训练、压力测试或奖励搜索，不能作为真实特技动作完成证据。
- 生成后的 spec 会被 `create_asap_evolution_configs.py` 和 `create_asap_task_profiles.py` 消费。
