"""CLI: skeleton alignment parameter study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install 'worldmix-skeleton[study]'") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeleton alignment parameter study")
    parser.add_argument("--config", default="experiments/configs/character_alignment_study.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--glb", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    output_root = Path(args.output or cfg.get("output_root", "outputs/alignment_study"))
    output_root.mkdir(parents=True, exist_ok=True)

    from worldmix.eval.character_study import run_character_parameter_study

    glb = Path(args.glb or cfg.get("glb", "outputs/model_diane/shape.glb"))
    if not glb.exists():
        raise FileNotFoundError(f"GLB not found: {glb}")

    summary = run_character_parameter_study(
        glb,
        output_root,
        ablations=cfg.get("ablations"),
        frames=int(cfg.get("frames", 8)),
    )
    report = {
        "scope": "skeleton_animation_only",
        "config": str(cfg_path.resolve()) if cfg_path.exists() else None,
        "glb": str(glb.resolve()),
        **summary,
    }
    report_path = output_root / "alignment_study_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== Skeleton alignment study ===")
    print(f"table: {summary['table_csv']}")
    print(
        f"best_joint: {summary['best_by_joint_error']['name']} "
        f"(err={summary['best_by_joint_error']['mean_joint_error']:.4f})"
    )
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
