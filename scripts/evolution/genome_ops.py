"""Initialization, crossover, and mutation for structured algorithm genomes."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

from schemas import (
    BASELINE_GENOME,
    AlgorithmGenome,
    DomainRandomizationGenes,
    GenomeMetadata,
    PPOGenes,
    ResourceGenes,
    RewardGenes,
    SamplingGenes,
    TerminationGenes,
)
from validator import require_valid_genome


SECTION_CLASSES = {
    "reward": RewardGenes,
    "sampling": SamplingGenes,
    "termination": TerminationGenes,
    "ppo": PPOGenes,
    "domain_randomization": DomainRandomizationGenes,
    "resource": ResourceGenes,
}

INT_KEYS = {
    "sampling.adaptive_kernel_size",
    "sampling.fixed_start_time_steps",
    "ppo.num_learning_epochs",
    "ppo.num_mini_batches",
    "resource.num_envs",
    "resource.stage1_iterations",
    "resource.stage2_iterations",
    "resource.full_iterations",
    "resource.save_interval",
    "resource.stage1_eval_episodes",
    "resource.stage2_eval_episodes",
    "resource.final_eval_episodes",
}

NETWORK_CHOICES = [
    [256, 256, 128],
    [512, 256, 128],
    [512, 512, 256],
    [1024, 512, 256],
]


def _sample_value(rng: random.Random, dotted: str, current: Any, config: dict[str, Any]) -> Any:
    search_space = config.get("search_space", {})
    if dotted not in search_space:
        if dotted.endswith("actor_hidden_dims") or dotted.endswith("critic_hidden_dims"):
            return list(rng.choice(NETWORK_CHOICES))
        if dotted.endswith("activation"):
            return rng.choice(["elu", "relu", "tanh"])
        return copy.deepcopy(current)
    lo, hi = search_space[dotted]
    if dotted in INT_KEYS:
        value = rng.randint(int(lo), int(hi))
        if dotted == "sampling.adaptive_kernel_size" and value % 2 == 0:
            value = min(int(hi), value + 1) if value + 1 <= int(hi) else max(int(lo), value - 1)
        return value
    return float(rng.uniform(float(lo), float(hi)))


def _section_dict(section: str, genome: AlgorithmGenome) -> dict[str, Any]:
    return asdict(getattr(genome, section))


def _with_sections(
    metadata: GenomeMetadata,
    sections: dict[str, dict[str, Any]],
    rationale: list[str] | None = None,
) -> AlgorithmGenome:
    return AlgorithmGenome(
        metadata=metadata,
        reward=RewardGenes(**sections["reward"]),
        sampling=SamplingGenes(**sections["sampling"]),
        termination=TerminationGenes(**sections["termination"]),
        ppo=PPOGenes(**sections["ppo"]),
        domain_randomization=DomainRandomizationGenes(**sections["domain_randomization"]),
        resource=ResourceGenes(**sections["resource"]),
        rationale=rationale or [],
    )


def seed_population(config: dict[str, Any], population_size: int, generation: int = 0) -> list[AlgorithmGenome]:
    """Create deterministic local seed genomes without calling an LLM."""

    rng = random.Random(int(config.get("evolution", {}).get("random_seed", 0)) + generation)
    population: list[AlgorithmGenome] = []

    baseline = copy.deepcopy(BASELINE_GENOME)
    baseline.metadata.genome_id = f"gen{generation}_baseline_000"
    baseline.metadata.generation = generation
    baseline.rationale = ["当前仓库已验证的 G1 50cm 膝爬 BeyondMimic 基线。"]
    baseline.resource.num_envs = int(config["resource_defaults"]["num_envs"])
    baseline.resource.stage1_iterations = int(config["resource_defaults"]["stage1_iterations"])
    baseline.resource.stage2_iterations = int(config["resource_defaults"]["stage2_iterations"])
    baseline.resource.full_iterations = int(config["resource_defaults"]["full_iterations"])
    baseline.resource.save_interval = int(config["resource_defaults"]["save_interval"])
    baseline.resource.stage1_eval_episodes = int(config["resource_defaults"]["stage1_eval_episodes"])
    baseline.resource.stage2_eval_episodes = int(config["resource_defaults"]["stage2_eval_episodes"])
    baseline.resource.final_eval_episodes = int(config["resource_defaults"]["final_eval_episodes"])
    require_valid_genome(baseline, config)
    population.append(baseline)

    templates = [
        {
            "description": "增强 motion-start 覆盖，并轻微放宽接触型终止，降低起始阶段失败。",
            "sampling.fixed_start_probability": 0.98,
            "sampling.fixed_start_time_steps": 0,
            "sampling.adaptive_uniform_ratio": 1.15,
            "termination.ee_body_pos_z_threshold": 0.34,
            "termination.anchor_pos_z_threshold": 0.32,
        },
        {
            "description": "强化身体相对位姿跟踪，降低速度项权重，减少越障接触阶段的姿态漂移。",
            "reward.motion_body_pos_weight": 1.35,
            "reward.motion_body_pos_std": 0.24,
            "reward.motion_body_ori_weight": 1.15,
            "reward.motion_body_lin_vel_weight": 0.75,
            "reward.motion_body_ang_vel_weight": 0.75,
        },
        {
            "description": "增加探索和较宽 KL，服务翻墙/钻洞前期动作变体搜索。",
            "ppo.entropy_coef": 0.009,
            "ppo.desired_kl": 0.016,
            "ppo.clip_param": 0.24,
            "ppo.learning_rate": 0.00075,
        },
        {
            "description": "增加摩擦和质心随机化，为后续 sim2real 预留鲁棒性。",
            "domain_randomization.friction_static_min": 0.45,
            "domain_randomization.friction_static_max": 1.8,
            "domain_randomization.friction_dynamic_min": 0.4,
            "domain_randomization.friction_dynamic_max": 1.35,
            "domain_randomization.joint_default_pos_abs": 0.02,
            "domain_randomization.torso_com_x_abs": 0.035,
        },
    ]

    for index in range(1, population_size):
        sections = {name: _section_dict(name, baseline) for name in SECTION_CLASSES}
        if index - 1 < len(templates):
            template = templates[index - 1]
            description = str(template["description"])
            for dotted, value in template.items():
                if dotted == "description":
                    continue
                section, key = dotted.split(".", 1)
                sections[section][key] = value
        else:
            description = "随机采样的任务自适应 BeyondMimic 候选。"
            for section, values in sections.items():
                for key, current in list(values.items()):
                    dotted = f"{section}.{key}"
                    if rng.random() < 0.35:
                        values[key] = _sample_value(rng, dotted, current, config)

        metadata = GenomeMetadata(
            genome_id=f"gen{generation}_local_{index:03d}",
            generation=generation,
            parent_ids=[baseline.metadata.genome_id],
            created_by="local-seed",
            description=description,
        )
        genome = _with_sections(
            metadata,
            sections,
            rationale=[
                description,
                "该候选由本地受控搜索空间生成，可作为 Mimimax M3 候选失败时的 fallback。",
            ],
        )
        require_valid_genome(genome, config)
        population.append(genome)

    return population[:population_size]


def crossover(
    parent_a: AlgorithmGenome,
    parent_b: AlgorithmGenome,
    child_id: str,
    generation: int,
    rng: random.Random,
) -> AlgorithmGenome:
    sections: dict[str, dict[str, Any]] = {}
    for section in SECTION_CLASSES:
        values_a = _section_dict(section, parent_a)
        values_b = _section_dict(section, parent_b)
        sections[section] = {
            key: copy.deepcopy(values_a[key] if rng.random() < 0.5 else values_b[key]) for key in values_a
        }
    metadata = GenomeMetadata(
        genome_id=child_id,
        generation=generation,
        parent_ids=[parent_a.metadata.genome_id, parent_b.metadata.genome_id],
        created_by="crossover",
        description="由两名高分候选交叉得到的 BeyondMimic 任务适配候选。",
    )
    return _with_sections(metadata, sections, rationale=["交叉保留高分候选的 reward、采样和 PPO 局部结构。"])


def mutate(genome: AlgorithmGenome, config: dict[str, Any], mutation_rate: float, rng: random.Random) -> AlgorithmGenome:
    mutated = copy.deepcopy(genome)
    for section in SECTION_CLASSES:
        values = _section_dict(section, mutated)
        for key, current in values.items():
            dotted = f"{section}.{key}"
            if rng.random() < mutation_rate:
                values[key] = _sample_value(rng, dotted, current, config)
        setattr(mutated, section, SECTION_CLASSES[section](**values))
    mutated.rationale.append(f"mutation_rate={mutation_rate:.3f} 的本地安全变异。")
    return mutated


def next_generation(
    ranked_parents: list[AlgorithmGenome],
    config: dict[str, Any],
    population_size: int,
    generation: int,
) -> list[AlgorithmGenome]:
    """Build the next generation from ranked parents."""

    if not ranked_parents:
        return seed_population(config, population_size, generation=generation)

    rng = random.Random(int(config.get("evolution", {}).get("random_seed", 0)) + generation * 1009)
    elite_count = max(1, int(config.get("evolution", {}).get("elite_count", 1)))
    mutation_rate = float(config.get("evolution", {}).get("mutation_rate", 0.25))

    children: list[AlgorithmGenome] = []
    for elite in ranked_parents[:elite_count]:
        kept = copy.deepcopy(elite)
        kept.metadata.generation = generation
        kept.metadata.genome_id = f"gen{generation}_elite_{len(children):03d}"
        kept.metadata.created_by = "elite-carry"
        children.append(kept)

    while len(children) < population_size:
        parent_a = rng.choice(ranked_parents[: max(2, min(len(ranked_parents), elite_count + 2))])
        parent_b = rng.choice(ranked_parents[: max(2, min(len(ranked_parents), elite_count + 2))])
        child = crossover(parent_a, parent_b, f"gen{generation}_cross_{len(children):03d}", generation, rng)
        child = mutate(child, config, mutation_rate, rng)
        require_valid_genome(child, config)
        children.append(child)

    return children[:population_size]

