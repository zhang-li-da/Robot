from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1KneeClimbPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_knee_climb"


@configclass
class G1JumpLeapPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_jump_leap"


@configclass
class G1BackflipPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_backflip"


@configclass
class G1WallTurnPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_wall_turn"


@configclass
class G1CrawlTunnelPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_crawl_tunnel"


@configclass
class G1RollVaultPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_roll_vault"


@configclass
class G1RollVaultV2PPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_roll_vault_v2"


@configclass
class G1RollVaultV3PPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_roll_vault_v3"


@configclass
class G1RollVaultV4PPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_roll_vault_v4"


@configclass
class G1DiveRollPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_dive_roll"


@configclass
class G1DiveRollV2PPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_dive_roll_v2"


@configclass
class G1DiveRollV3PPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "g1_dive_roll_v3"
