"""Structured genome schema for task-adaptive BeyondMimic evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenomeMetadata:
    genome_id: str
    generation: int
    parent_ids: list[str] = field(default_factory=list)
    task_family: str = "humanoid_obstacle_stunt"
    created_by: str = "local"
    description: str = ""


@dataclass
class RewardGenes:
    motion_global_anchor_pos_weight: float = 0.5
    motion_global_anchor_pos_std: float = 0.3
    motion_global_anchor_ori_weight: float = 0.5
    motion_global_anchor_ori_std: float = 0.4
    motion_body_pos_weight: float = 1.0
    motion_body_pos_std: float = 0.3
    motion_body_ori_weight: float = 1.0
    motion_body_ori_std: float = 0.4
    motion_body_lin_vel_weight: float = 1.0
    motion_body_lin_vel_std: float = 1.0
    motion_body_ang_vel_weight: float = 1.0
    motion_body_ang_vel_std: float = 3.14
    action_rate_l2_weight: float = -0.1
    joint_limit_weight: float = -10.0
    undesired_contacts_weight: float = -0.1
    task_progress_weight: float = 0.0
    clearance_weight: float = 0.0


@dataclass
class SamplingGenes:
    adaptive_uniform_ratio: float = 1.0
    adaptive_kernel_size: int = 5
    adaptive_lambda: float = 0.8
    adaptive_alpha: float = 0.001
    fixed_start_probability: float = 0.95
    fixed_start_time_steps: int = 0


@dataclass
class TerminationGenes:
    anchor_pos_z_threshold: float = 0.25
    anchor_ori_threshold: float = 0.8
    ee_body_pos_z_threshold: float = 0.25


@dataclass
class PPOGenes:
    learning_rate: float = 0.001
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"


@dataclass
class DomainRandomizationGenes:
    friction_static_min: float = 0.3
    friction_static_max: float = 1.6
    friction_dynamic_min: float = 0.3
    friction_dynamic_max: float = 1.2
    joint_default_pos_abs: float = 0.01
    torso_com_x_abs: float = 0.025
    torso_com_y_abs: float = 0.05
    torso_com_z_abs: float = 0.05
    push_interval_min: float = 1.0
    push_interval_max: float = 3.0


@dataclass
class ResourceGenes:
    num_envs: int = 2048
    stage1_iterations: int = 800
    stage2_iterations: int = 1800
    full_iterations: int = 4000
    save_interval: int = 400
    stage1_eval_episodes: int = 16
    stage2_eval_episodes: int = 32
    final_eval_episodes: int = 64


@dataclass
class AlgorithmGenome:
    metadata: GenomeMetadata
    reward: RewardGenes = field(default_factory=RewardGenes)
    sampling: SamplingGenes = field(default_factory=SamplingGenes)
    termination: TerminationGenes = field(default_factory=TerminationGenes)
    ppo: PPOGenes = field(default_factory=PPOGenes)
    domain_randomization: DomainRandomizationGenes = field(default_factory=DomainRandomizationGenes)
    resource: ResourceGenes = field(default_factory=ResourceGenes)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlgorithmGenome":
        required = [
            "metadata",
            "reward",
            "sampling",
            "termination",
            "ppo",
            "domain_randomization",
            "resource",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Genome is missing required sections: {missing}")

        return cls(
            metadata=GenomeMetadata(**data["metadata"]),
            reward=RewardGenes(**data["reward"]),
            sampling=SamplingGenes(**data["sampling"]),
            termination=TerminationGenes(**data["termination"]),
            ppo=PPOGenes(**data["ppo"]),
            domain_randomization=DomainRandomizationGenes(**data["domain_randomization"]),
            resource=ResourceGenes(**data["resource"]),
            rationale=list(data.get("rationale", [])),
        )


BASELINE_GENOME = AlgorithmGenome(
    metadata=GenomeMetadata(
        genome_id="baseline_g1_knee_climb",
        generation=0,
        parent_ids=[],
        created_by="manual-baseline",
        description="Current G1 50cm knee-climb BeyondMimic baseline.",
    )
)


SECTION_TYPES = {
    "reward": RewardGenes,
    "sampling": SamplingGenes,
    "termination": TerminationGenes,
    "ppo": PPOGenes,
    "domain_randomization": DomainRandomizationGenes,
    "resource": ResourceGenes,
}


def section_items(genome: AlgorithmGenome) -> list[tuple[str, str, Any]]:
    items: list[tuple[str, str, Any]] = []
    for section in SECTION_TYPES:
        section_obj = getattr(genome, section)
        for key, value in asdict(section_obj).items():
            items.append((section, key, value))
    return items

