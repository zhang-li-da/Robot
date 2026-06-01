"""Validation for structured BeyondMimic evolution genomes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from schemas import AlgorithmGenome


INTEGER_FIELDS = {
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

ALLOWED_ACTIVATIONS = {"elu", "relu", "tanh"}
ALLOWED_HIDDEN_DIMS = {
    (256, 256, 128),
    (512, 256, 128),
    (512, 512, 256),
    (1024, 512, 256),
}


class GenomeValidationError(ValueError):
    """Raised when a genome violates schema, ranges, or task invariants."""


def _range_for(config: dict[str, Any], dotted_name: str) -> tuple[float, float] | None:
    value = config.get("search_space", {}).get(dotted_name)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise GenomeValidationError(f"Invalid search_space range for {dotted_name}: {value}")
    return float(value[0]), float(value[1])


def _validate_number(config: dict[str, Any], dotted_name: str, value: Any, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{dotted_name} must be numeric, got {type(value).__name__}")
        return
    if dotted_name in INTEGER_FIELDS and int(value) != value:
        errors.append(f"{dotted_name} must be an integer, got {value}")
    allowed = _range_for(config, dotted_name)
    if allowed is None:
        return
    lo, hi = allowed
    if not lo <= float(value) <= hi:
        errors.append(f"{dotted_name}={value} outside [{lo}, {hi}]")


def validate_genome(genome: AlgorithmGenome, config: dict[str, Any]) -> list[str]:
    """Return a list of validation errors. An empty list means the genome is accepted."""

    errors: list[str] = []
    genome_dict = genome.to_dict()
    for section in ["reward", "sampling", "termination", "ppo", "domain_randomization", "resource"]:
        for key, value in genome_dict[section].items():
            dotted = f"{section}.{key}"
            if isinstance(value, list):
                continue
            if dotted == "ppo.activation":
                continue
            _validate_number(config, dotted, value, errors)

    ppo = genome.ppo
    if ppo.activation not in ALLOWED_ACTIVATIONS:
        errors.append(f"ppo.activation must be one of {sorted(ALLOWED_ACTIVATIONS)}, got {ppo.activation}")
    for name, dims in [("actor_hidden_dims", ppo.actor_hidden_dims), ("critic_hidden_dims", ppo.critic_hidden_dims)]:
        if tuple(dims) not in ALLOWED_HIDDEN_DIMS:
            errors.append(f"ppo.{name}={dims} is not in the allowed network set")

    sampling = genome.sampling
    if sampling.adaptive_kernel_size < 1:
        errors.append("sampling.adaptive_kernel_size must be positive")
    if sampling.adaptive_kernel_size % 2 == 0:
        errors.append("sampling.adaptive_kernel_size should be odd to keep phase smoothing centered")
    if sampling.fixed_start_probability < 0.25:
        errors.append("sampling.fixed_start_probability must preserve motion-start training coverage")
    if sampling.fixed_start_time_steps > 30:
        errors.append("sampling.fixed_start_time_steps is too wide for strict motion-start evaluation")

    reward = genome.reward
    if reward.action_rate_l2_weight >= 0.0:
        errors.append("reward.action_rate_l2_weight must stay negative")
    if reward.joint_limit_weight >= 0.0:
        errors.append("reward.joint_limit_weight must stay negative")
    if reward.undesired_contacts_weight >= 0.0:
        errors.append("reward.undesired_contacts_weight must stay negative")

    dr = genome.domain_randomization
    if dr.friction_static_min > dr.friction_static_max:
        errors.append("domain_randomization static friction min cannot exceed max")
    if dr.friction_dynamic_min > dr.friction_dynamic_max:
        errors.append("domain_randomization dynamic friction min cannot exceed max")
    if dr.push_interval_min > dr.push_interval_max:
        errors.append("domain_randomization push interval min cannot exceed max")

    resource = genome.resource
    if resource.stage1_iterations > resource.stage2_iterations:
        errors.append("resource.stage1_iterations cannot exceed stage2_iterations")
    if resource.stage2_iterations > resource.full_iterations:
        errors.append("resource.stage2_iterations cannot exceed full_iterations")
    minimum_final_trials = int(config.get("evolution", {}).get("minimum_final_trials", 50))
    if resource.final_eval_episodes < minimum_final_trials:
        errors.append(f"resource.final_eval_episodes must be >= {minimum_final_trials}")

    metadata = genome.metadata
    if not metadata.genome_id:
        errors.append("metadata.genome_id cannot be empty")
    if "/" in metadata.genome_id or ".." in metadata.genome_id:
        errors.append("metadata.genome_id cannot contain path separators")

    # Keep only JSON-serializable primitive fields; this catches accidental object injection.
    for section, values in genome_dict.items():
        if section == "rationale":
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                errors.append("rationale must be a list of strings")
            continue
        if section == "metadata":
            continue
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                continue
            if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                continue
            errors.append(f"{section}.{key} is not a primitive JSON field")

    return errors


def require_valid_genome(genome: AlgorithmGenome, config: dict[str, Any]) -> None:
    errors = validate_genome(genome, config)
    if errors:
        raise GenomeValidationError("; ".join(errors))


def flatten_genome(genome: AlgorithmGenome) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in ["reward", "sampling", "termination", "ppo", "domain_randomization", "resource"]:
        for key, value in asdict(getattr(genome, section)).items():
            flat[f"{section}.{key}"] = value
    return flat
