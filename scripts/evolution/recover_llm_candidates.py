"""Recover complete candidate genomes from a truncated LLM raw-text response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from genome_ops import normalize_genome_for_config, seed_population  # noqa: E402
from minimax_client import _extract_json_object  # noqa: E402
from planner import write_plan_files  # noqa: E402
from run_generation import parse_llm_candidates, write_genomes  # noqa: E402
from validator import validate_genome  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover valid candidates from llm_raw_text.txt.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw_text", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--fill_with_local", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    raw_text = args.raw_text.read_text(encoding="utf-8")
    payload = _extract_json_object(raw_text)
    genomes = parse_llm_candidates(payload, config, args.population_size)
    if args.fill_with_local and len(genomes) < args.population_size:
        existing_ids = {genome.metadata.genome_id for genome in genomes}
        for genome in seed_population(config, args.population_size, generation=args.generation):
            if genome.metadata.genome_id in existing_ids:
                continue
            genomes.append(genome)
            if len(genomes) >= args.population_size:
                break

    accepted = []
    validation_report = []
    for genome in genomes[: args.population_size]:
        genome = normalize_genome_for_config(genome, config)
        errors = validate_genome(genome, config)
        validation_report.append({"genome_id": genome.metadata.genome_id, "valid": not errors, "errors": errors})
        if not errors:
            accepted.append(genome)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(
            {
                "raw_text": str(args.raw_text),
                "parse_repair": payload.get("parse_repair", {}),
                "population_requested": args.population_size,
                "population_recovered": len(accepted),
                "genome_ids": [genome.metadata.genome_id for genome in accepted],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_genomes(accepted, args.output_dir)
    write_plan_files(accepted, config, args.output_dir)
    print((args.output_dir / "recovery_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
