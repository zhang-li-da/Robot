import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# The knee-climb source dataset provides a 50 cm obstacle mesh.  After applying
# the same raw(-Y)->IsaacLab(+X) transform used by the motion converter, the
# mesh's 50 cm top surface spans roughly:
#   x=[1.0171, 1.6515], y=[-0.3389, 0.5306], z=[0.0, 0.5087].
OBSTACLE_SIZE = (0.6344, 0.8695, 0.5087)
OBSTACLE_CENTER = (1.3343, 0.0959, 0.25435)


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
