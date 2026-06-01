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

from genome_ops import seed_population
from minimax_client import MimimaxClientError, generate_candidates, load_credentials
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
    parser.add_argument("--llm_timeout", type=float, default=90.0, help="Mimimax request timeout in seconds.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"scores": []}
    return load_json(path)


def render_prompt(config: dict[str, Any], history: dict[str, Any]) -> str:
    template_path = Path(config["llm"]["prompt_template"])
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{CONFIG_JSON}}", json.dumps(config, indent=2, ensure_ascii=False))
        .replace("{{HISTORY_JSON}}", json.dumps(history, indent=2, ensure_ascii=False))
    )


def parse_llm_candidates(payload: dict[str, Any], config: dict[str, Any], population_size: int) -> list[AlgorithmGenome]:
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
    population_size = int(args.population_size or config.get("evolution", {}).get("population_size", 4))
    history = load_history(args.history)

    run_id = datetime.now().strftime(f"%Y%m%d_%H%M%S_gen{args.generation:02d}")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "history_snapshot.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    genomes: list[AlgorithmGenome] = []
    if args.use_llm:
        prompt = render_prompt(config, history)
        prompt_path = output_dir / "prompt_rendered.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        try:
            credentials = load_credentials(config)
            print(
                "[INFO] Mimimax credentials loaded: "
                f"url={credentials.api_url}, mode={credentials.api_mode}, model={credentials.model}, "
                f"key={credentials.redacted_key}"
            )
            payload = generate_candidates(prompt, config, credentials, timeout=args.llm_timeout)
            (output_dir / "llm_response.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            genomes = parse_llm_candidates(payload, config, population_size)
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
