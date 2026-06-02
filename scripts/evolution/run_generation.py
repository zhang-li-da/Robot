"""Generate one population of task-adaptive BeyondMimic genomes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from genome_ops import normalize_genome_for_config, seed_population
from minimax_client import MimimaxClientError, MimimaxJSONError, generate_candidates, load_credentials
from planner import write_plan_files
from schemas import AlgorithmGenome
from validator import validate_genome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate task-adaptive BeyondMimic evolution candidates.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_root", default="outputs/evolution", type=Path)
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--use_llm", action="store_true", help="Call Mimimax M3 for candidate generation.")
    parser.add_argument("--dry_run", action="store_true", help="Generate plans without executing training.")
    parser.add_argument("--history", type=Path, default=None, help="Optional scoreboard/history JSON.")
    parser.add_argument("--feedback", type=Path, default=None, help="Optional structured feedback JSON from feedback_analyzer.py.")
    parser.add_argument("--llm_timeout", type=float, default=None, help="Mimimax request timeout in seconds.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"scores": []}
    return load_json(path)


def load_motion_catalog(config: dict[str, Any]) -> dict[str, Any]:
    catalog_path = config.get("task", {}).get("motion_catalog") or config.get("motion_catalog")
    if not catalog_path:
        return {}
    path = Path(str(catalog_path))
    if not path.exists():
        return {"missing_motion_catalog": str(path)}
    catalog = load_json(path)
    task_name = str(config.get("task", {}).get("name", ""))
    filter_tasks = config.get("task", {}).get("motion_catalog_filter_tasks") or ([task_name] if task_name else [])
    clips = catalog.get("clips", [])
    if isinstance(clips, list) and filter_tasks:
        task_clips = [
            item
            for item in clips
            if any(filter_task in item.get("suggested_tasks", []) for filter_task in filter_tasks)
        ]
        if task_clips:
            catalog = dict(catalog)
            catalog["clips"] = task_clips[:20]
            catalog["prompt_filter"] = {
                "task_name": task_name,
                "filter_tasks": filter_tasks,
                "matched_clips": len(task_clips),
                "included": len(catalog["clips"]),
            }
    return catalog


def load_task_profile(config: dict[str, Any]) -> dict[str, Any]:
    profile_path = config.get("task", {}).get("task_feature_profile") or config.get("task_feature_profile")
    if not profile_path:
        return {}
    path = Path(str(profile_path))
    if not path.exists():
        return {"missing_task_feature_profile": str(path)}
    return load_json(path)


def load_asset_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = config.get("task", {}).get("asset_manifest") or config.get("asset_manifest")
    if not manifest_path:
        return {}
    path = Path(str(manifest_path))
    if not path.exists():
        return {"missing_asset_manifest": str(path)}
    manifest = load_json(path)
    return {
        "schema_version": manifest.get("schema_version"),
        "asap_root": manifest.get("asap_root"),
        "purpose": manifest.get("purpose"),
        "counts": manifest.get("counts", {}),
        "known_limitations": manifest.get("known_limitations", []),
        "tag_counts": manifest.get("tag_counts", {}),
        "sim2real_mimic_models": manifest.get("sim2real_mimic_models", []),
        "sim2real_locomotion_models": manifest.get("sim2real_locomotion_models", []),
        "algorithm_priors": manifest.get("algorithm_priors", {}),
    }


def load_algorithm_priors(config: dict[str, Any]) -> dict[str, Any]:
    priors_path = config.get("task", {}).get("algorithm_priors") or config.get("algorithm_priors")
    if not priors_path:
        priors_path = "evolution/algorithm_priors/asap_algorithm_priors.json"
    path = Path(str(priors_path))
    if not path.exists():
        return {"missing_algorithm_priors": str(path)}
    priors = load_json(path)
    return {
        "schema_version": priors.get("schema_version"),
        "asap_root": priors.get("asap_root"),
        "purpose": priors.get("purpose"),
        "priors": priors.get("priors", {}),
        "task_family_guidance": priors.get("task_family_guidance", {}),
        "llm_constraints": priors.get("llm_constraints", []),
    }


def render_prompt(config: dict[str, Any], history: dict[str, Any], population_size: int) -> str:
    template_path = Path(config["llm"]["prompt_template"])
    template = template_path.read_text(encoding="utf-8")
    motion_catalog = load_motion_catalog(config)
    task_profile = load_task_profile(config)
    asset_manifest = load_asset_manifest(config)
    algorithm_priors = load_algorithm_priors(config)
    return (
        template.replace("{{CONFIG_JSON}}", json.dumps(config, indent=2, ensure_ascii=False))
        .replace("{{TASK_PROFILE_JSON}}", json.dumps(task_profile, indent=2, ensure_ascii=False))
        .replace("{{HISTORY_JSON}}", json.dumps(history, indent=2, ensure_ascii=False))
        .replace("{{MOTION_CATALOG_JSON}}", json.dumps(motion_catalog, indent=2, ensure_ascii=False))
        .replace("{{ASSET_MANIFEST_JSON}}", json.dumps(asset_manifest, indent=2, ensure_ascii=False))
        .replace("{{ALGORITHM_PRIORS_JSON}}", json.dumps(algorithm_priors, indent=2, ensure_ascii=False))
        .replace("{{REQUESTED_POPULATION_SIZE}}", str(population_size))
    )


def render_prompt_with_feedback(
    config: dict[str, Any],
    history: dict[str, Any],
    feedback: dict[str, Any],
    population_size: int,
) -> str:
    return render_prompt(config, history, population_size).replace(
        "{{FEEDBACK_JSON}}",
        json.dumps(feedback, indent=2, ensure_ascii=False),
    )


def _context_baseline_success(history: dict[str, Any], feedback: dict[str, Any]) -> float:
    candidates = [
        history.get("baseline", {}),
        feedback.get("baseline", {}),
        feedback.get("population_feedback", {}),
    ]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("success_rate", "baseline_success_rate"):
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            return value
    return 0.0


def _clip_context_value(config: dict[str, Any], dotted: str, value: float) -> float:
    bounds = config.get("search_space", {}).get(dotted)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return value
    lo, hi = float(bounds[0]), float(bounds[1])
    return max(lo, min(hi, value))


def _apply_high_baseline_guard(
    genome: AlgorithmGenome,
    config: dict[str, Any],
    history: dict[str, Any],
    feedback: dict[str, Any],
) -> AlgorithmGenome:
    """Prevent gen0 high-baseline proxy tasks from drifting into from-scratch search."""

    guarded = genome
    baseline_success = _context_baseline_success(history, feedback)
    if baseline_success < 0.90:
        return guarded

    target_x = float(config.get("task", {}).get("target_x", 1.0))
    proxy_note = str(config.get("task", {}).get("success_criteria", {}).get("proxy_note", ""))

    guarded.sampling.fixed_start_probability = _clip_context_value(
        config,
        "sampling.fixed_start_probability",
        max(float(guarded.sampling.fixed_start_probability), 0.82),
    )
    guarded.termination.anchor_pos_z_threshold = _clip_context_value(
        config,
        "termination.anchor_pos_z_threshold",
        max(float(guarded.termination.anchor_pos_z_threshold), 0.30),
    )
    guarded.termination.ee_body_pos_z_threshold = _clip_context_value(
        config,
        "termination.ee_body_pos_z_threshold",
        max(float(guarded.termination.ee_body_pos_z_threshold), 0.32),
    )
    guarded.reward.phase_progress_weight = _clip_context_value(
        config,
        "reward.phase_progress_weight",
        max(float(guarded.reward.phase_progress_weight), 0.25),
    )
    if target_x <= 0.10 or proxy_note:
        guarded.reward.task_progress_weight = _clip_context_value(
            config,
            "reward.task_progress_weight",
            min(float(guarded.reward.task_progress_weight), 0.35),
        )
    note = "高成功率baseline保护：候选必须贴近baseline并避免proxy退化"
    if note not in guarded.rationale:
        guarded.rationale = list(guarded.rationale) + [note]
    return guarded


def _normalize_with_context(
    genome: AlgorithmGenome,
    config: dict[str, Any],
    history: dict[str, Any],
    feedback: dict[str, Any],
) -> AlgorithmGenome:
    genome = normalize_genome_for_config(genome, config)
    genome = _apply_high_baseline_guard(genome, config, history, feedback)
    return normalize_genome_for_config(genome, config)


def parse_llm_candidates(
    payload: dict[str, Any],
    config: dict[str, Any],
    population_size: int,
    history: dict[str, Any],
    feedback: dict[str, Any],
) -> list[AlgorithmGenome]:
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Mimimax payload must contain a candidates list")

    accepted: list[AlgorithmGenome] = []
    for index, raw in enumerate(raw_candidates):
        try:
            genome = AlgorithmGenome.from_dict(raw)
        except (TypeError, ValueError) as exc:
            print(f"[WARN] reject candidate[{index}] schema error: {exc}")
            continue
        genome = _normalize_with_context(genome, config, history, feedback)
        errors = validate_genome(genome, config)
        if errors:
            print(f"[WARN] reject {genome.metadata.genome_id}: {'; '.join(errors)}")
            continue
        accepted.append(genome)
        if len(accepted) >= population_size:
            break
    return accepted


def write_genomes(genomes: list[AlgorithmGenome], output_dir: Path) -> None:
    genomes_dir = output_dir / "genomes"
    genomes_dir.mkdir(parents=True, exist_ok=True)
    for genome in genomes:
        path = genomes_dir / f"{genome.metadata.genome_id}.json"
        path.write_text(json.dumps(genome.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    llm_timeout = float(args.llm_timeout or config.get("llm", {}).get("timeout_seconds", 300.0))
    population_size = int(args.population_size or config.get("evolution", {}).get("population_size", 4))
    history = load_history(args.history)
    feedback = load_json(args.feedback) if args.feedback is not None and args.feedback.exists() else {}
    motion_catalog = load_motion_catalog(config)

    run_id_base = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_gen{args.generation:02d}")
    output_dir = args.output_root / run_id_base
    suffix = 1
    while output_dir.exists():
        output_dir = args.output_root / f"{run_id_base}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "history_snapshot.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if feedback:
        (output_dir / "feedback_snapshot.json").write_text(
            json.dumps(feedback, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if motion_catalog:
        (output_dir / "motion_catalog_snapshot.json").write_text(
            json.dumps(motion_catalog, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    asset_manifest = load_asset_manifest(config)
    if asset_manifest:
        (output_dir / "asset_manifest_snapshot.json").write_text(
            json.dumps(asset_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    task_profile = load_task_profile(config)
    if task_profile:
        (output_dir / "task_profile_snapshot.json").write_text(
            json.dumps(task_profile, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    genomes: list[AlgorithmGenome] = []
    if args.use_llm:
        prompt = render_prompt_with_feedback(config, history, feedback, population_size)
        prompt_path = output_dir / "prompt_rendered.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        try:
            credentials = load_credentials(config)
            print(
                "[INFO] Mimimax credentials loaded: "
                f"url={credentials.api_url}, mode={credentials.api_mode}, model={credentials.model}"
            )
            payload = generate_candidates(prompt, config, credentials, timeout=llm_timeout)
            (output_dir / "llm_response.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            genomes = parse_llm_candidates(payload, config, population_size, history, feedback)
        except MimimaxJSONError as exc:
            (output_dir / "llm_raw_text.txt").write_text(exc.raw_text, encoding="utf-8")
            if exc.raw_response:
                (output_dir / "llm_raw_response.json").write_text(
                    json.dumps(exc.raw_response, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            print(f"[WARN] Mimimax returned invalid JSON, falling back to local seeds: {exc}")
        except (MimimaxClientError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[WARN] Mimimax generation failed, falling back to local seeds: {exc}")

    if len(genomes) < population_size:
        fallback = seed_population(config, population_size, generation=args.generation)
        existing_ids = {genome.metadata.genome_id for genome in genomes}
        for genome in fallback:
            if genome.metadata.genome_id in existing_ids:
                continue
            genomes.append(genome)
            if len(genomes) >= population_size:
                break

    validation_report: list[dict[str, Any]] = []
    accepted: list[AlgorithmGenome] = []
    for genome in genomes:
        genome = _normalize_with_context(genome, config, history, feedback)
        errors = validate_genome(genome, config)
        validation_report.append({"genome_id": genome.metadata.genome_id, "valid": not errors, "errors": errors})
        if not errors:
            accepted.append(genome)
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_genomes(accepted, output_dir)
    write_plan_files(accepted, config, output_dir)

    summary = {
        "output_dir": str(output_dir),
        "generation": args.generation,
        "population_requested": population_size,
        "population_accepted": len(accepted),
        "use_llm": args.use_llm,
        "dry_run": args.dry_run,
        "genome_ids": [genome.metadata.genome_id for genome in accepted],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
