import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import RewardsCfg, TrackingEnvCfg


# The knee-climb source dataset provides a 50 cm obstacle mesh.  After applying
# the same raw(-Y)->IsaacLab(+X) transform used by the motion converter, the
# mesh's 50 cm top surface spans roughly:
#   x=[1.0171, 1.6515], y=[-0.3389, 0.5306], z=[0.0, 0.5087].
OBSTACLE_SIZE = (0.6344, 0.8695, 0.5087)
OBSTACLE_CENTER = (1.3343, 0.0959, 0.25435)


@configclass
class G1KneeClimbRewardsCfg(RewardsCfg):
    # Disabled by default so the reproduced baseline remains unchanged.
    # The evolution framework can enable these task-specific terms through
    # Hydra overrides when searching beyond generic motion imitation.
    task_progress = RewTerm(
        func=mdp.motion_anchor_progress,
        weight=0.0,
        params={"command_name": "motion", "target_x": 1.70, "min_x": 0.0, "max_reward": 1.0},
    )
    clearance = RewTerm(
        func=mdp.body_clearance_over_height,
        weight=0.0,
        params={
            "command_name": "motion",
            "obstacle_height": OBSTACLE_SIZE[2],
            "target_clearance": 0.20,
            "body_names": [
                "left_knee_link",
                "right_knee_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        },
    )


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1KneeClimbEnvCfg(G1FlatEnvCfg):
    rewards: G1KneeClimbRewardsCfg = G1KneeClimbRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 3.0
        self.episode_length_s = 11.0
        # The single long knee-climb clip is easy to over-focus on the current
        # adaptive failure bin. Keep stronger uniform phase coverage so the
        # policy remains executable from frame 0.
        self.commands.motion.adaptive_uniform_ratio = 1.0
        # The deployed rollout always starts at the first frame.  Keep a
        # persistent reference-state-init path at the beginning of the clip so
        # adaptive sampling cannot starve the start phase.
        self.commands.motion.fixed_start_probability = 0.95
        self.commands.motion.fixed_start_time_steps = 0

        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=OBSTACLE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.2,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=OBSTACLE_CENTER),
        )

        # Knee contact is task-relevant for this motion; keep the original
        # BeyondMimic contact penalty for other non-support links.
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            (
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                r"(?!left_knee_link$)(?!right_knee_link$).+$"
            )
        ]


@configclass
class G1BackflipRewardsCfg(RewardsCfg):
    apex_height = RewTerm(
        func=mdp.anchor_height_over_min,
        weight=0.0,
        params={"command_name": "motion", "min_height": 1.05, "target_margin": 0.35},
    )
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=0.0,
        params={
            "command_name": "motion",
            "landing_phase": 0.78,
            "lin_vel_std": 0.6,
            "ang_vel_std": 1.2,
            "upright_std": 0.45,
        },
    )
    contact_force = RewTerm(
        func=mdp.contact_force_violation,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]),
            "threshold": 700.0,
        },
    )


@configclass
class G1WallTurnRewardsCfg(RewardsCfg):
    task_progress = RewTerm(
        func=mdp.motion_anchor_progress,
        weight=0.0,
        params={"command_name": "motion", "target_x": 1.20, "min_x": 0.0, "max_reward": 1.0},
    )
    clearance = RewTerm(
        func=mdp.body_clearance_over_height,
        weight=0.0,
        params={
            "command_name": "motion",
            "obstacle_height": 0.90,
            "target_clearance": 0.25,
            "body_names": [
                "pelvis",
                "left_knee_link",
                "right_knee_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        },
    )
    yaw_alignment = RewTerm(
        func=mdp.target_yaw_alignment,
        weight=0.0,
        params={"command_name": "motion", "target_yaw": math.pi, "std": 0.65, "start_phase": 0.55},
    )
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=0.0,
        params={
            "command_name": "motion",
            "landing_phase": 0.72,
            "lin_vel_std": 0.8,
            "ang_vel_std": 1.5,
            "upright_std": 0.55,
        },
    )
    contact_force = RewTerm(
        func=mdp.contact_force_violation,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]),
            "threshold": 900.0,
        },
    )


@configclass
class G1CrawlTunnelRewardsCfg(RewardsCfg):
    task_progress = RewTerm(
        func=mdp.motion_anchor_progress,
        weight=0.0,
        params={"command_name": "motion", "target_x": 1.50, "min_x": 0.0, "max_reward": 1.0},
    )
    ceiling_clearance = RewTerm(
        func=mdp.body_below_ceiling,
        weight=0.0,
        params={
            "command_name": "motion",
            "ceiling_height": 0.85,
            "target_margin": 0.20,
            "min_x": 0.30,
            "max_x": 1.90,
            "body_names": ["pelvis", "torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link"],
        },
    )
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=0.0,
        params={
            "command_name": "motion",
            "landing_phase": 0.82,
            "lin_vel_std": 0.5,
            "ang_vel_std": 1.0,
            "upright_std": 0.50,
        },
    )


@configclass
class G1BackflipEnvCfg(G1FlatEnvCfg):
    rewards: G1BackflipRewardsCfg = G1BackflipRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 3.0
        self.episode_length_s = 6.0
        self.commands.motion.fixed_start_probability = 0.90
        self.commands.motion.fixed_start_time_steps = 8
        self.commands.motion.adaptive_uniform_ratio = 0.90
        self.commands.motion.pose_range["z"] = (-0.02, 0.02)
        self.commands.motion.velocity_range["z"] = (-0.4, 0.4)
        self.commands.motion.velocity_range["pitch"] = (-1.2, 1.2)
        self.terminations.anchor_ori.params["threshold"] = 1.40
        self.terminations.ee_body_pos.params["threshold"] = 0.45
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
        ]


@configclass
class G1WallTurnEnvCfg(G1FlatEnvCfg):
    rewards: G1WallTurnRewardsCfg = G1WallTurnRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 3.5
        self.episode_length_s = 8.0
        self.commands.motion.fixed_start_probability = 0.90
        self.commands.motion.fixed_start_time_steps = 5
        self.commands.motion.adaptive_uniform_ratio = 1.05
        self.terminations.anchor_ori.params["threshold"] = 1.25
        self.terminations.ee_body_pos.params["threshold"] = 0.45
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            (
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
            )
        ]

        self.scene.wall = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Wall",
            spawn=sim_utils.CuboidCfg(
                size=(0.18, 1.20, 0.90),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.3, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.36, 0.40)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.15, 0.0, 0.45)),
        )


@configclass
class G1CrawlTunnelEnvCfg(G1FlatEnvCfg):
    rewards: G1CrawlTunnelRewardsCfg = G1CrawlTunnelRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 3.0
        self.episode_length_s = 9.0
        self.commands.motion.fixed_start_probability = 0.90
        self.commands.motion.fixed_start_time_steps = 4
        self.commands.motion.adaptive_uniform_ratio = 1.10
        self.terminations.anchor_pos.params["threshold"] = 0.35
        self.terminations.anchor_ori.params["threshold"] = 1.10
        self.terminations.ee_body_pos.params["threshold"] = 0.38
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            (
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                r"(?!left_knee_link$)(?!right_knee_link$)"
                r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
            )
        ]

        self.scene.tunnel_roof = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TunnelRoof",
            spawn=sim_utils.CuboidCfg(
                size=(1.60, 1.10, 0.08),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.24, 0.26)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.10, 0.0, 0.89)),
        )
