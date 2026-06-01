"""Validate constrained algorithm patch proposals for LLM-assisted evolution."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA = Path("evolution/algorithm_patch_schema.json")
PATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{2,63}$")
PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an algorithm patch JSON proposal.")
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_json_primitive_tree(value: Any) -> bool:
    if isinstance(value, PRIMITIVE_TYPES):
        return True
    if isinstance(value, list):
        return all(_is_json_primitive_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_primitive_tree(item) for key, item in value.items())
    return False


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def validate_patch(patch: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "patch_id",
        "target_task_family",
        "motivation",
        "patch_type",
        "insertion_template",
        "parameters",
    ]
    for key in required:
        if key not in patch:
            errors.append(f"missing required field: {key}")

    patch_id = str(patch.get("patch_id", ""))
    if not PATCH_ID_RE.match(patch_id):
        errors.append("patch_id must be 3-64 chars and contain only letters, numbers, '_' or '-'")
    if "/" in patch_id or ".." in patch_id:
        errors.append("patch_id must not contain path separators")

    allowed_types = set(schema.get("allowed_patch_types", []))
    patch_type = str(patch.get("patch_type", ""))
    if patch_type not in allowed_types:
        errors.append(f"patch_type must be one of {sorted(allowed_types)}, got {patch_type}")

    templates = schema.get("templates", {})
    insertion_template = str(patch.get("insertion_template", ""))
    template_ids = {str(value.get("template_id")) for value in templates.values() if isinstance(value, dict)}
    if insertion_template not in template_ids:
        errors.append(f"insertion_template must be one of {sorted(template_ids)}, got {insertion_template}")

    parameters = patch.get("parameters", {})
    if not isinstance(parameters, dict):
        errors.append("parameters must be a JSON object")
    elif not _is_json_primitive_tree(parameters):
        errors.append("parameters must contain only JSON primitive/list/object values")

    motivation = _strings(patch.get("motivation", []))
    if not motivation:
        errors.append("motivation must contain at least one failure tag or task feature")
    if any(len(item) > 80 for item in motivation):
        errors.append("motivation entries should be short failure tags or task features")

    forbidden_text = json.dumps(patch, ensure_ascii=False).lower()
    forbidden_tokens = ["subprocess", "socket", "urllib", "requests", "open(", "eval(", "exec("]
    for token in forbidden_tokens:
        if token in forbidden_text:
            errors.append(f"forbidden token in patch proposal: {token}")

    safety_checks = _strings(patch.get("safety_checks", []))
    required_safety = ["smoke", "validation"]
    safety_blob = " ".join(safety_checks).lower()
    for keyword in required_safety:
        if keyword not in safety_blob:
            errors.append(f"safety_checks should mention {keyword}")

    return errors


def main() -> int:
    args = parse_args()
    schema = load_json(args.schema)
    patch = load_json(args.patch)
    errors = validate_patch(patch, schema)
    payload = {
        "patch": str(args.patch),
        "valid": not errors,
        "errors": errors,
        "patch_id": patch.get("patch_id"),
        "patch_type": patch.get("patch_type"),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
