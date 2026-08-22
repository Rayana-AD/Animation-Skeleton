"""Parameter study: ViT vs geometric align, temperature, k_bones, motion command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldmix.character.glb_skeleton_fit import compute_vertex_skin_weights, segment_mesh_vertices
from worldmix.character.vit_align import align_glb_to_standard_skeleton
from worldmix.eval.metrics import (
    character_alignment_metrics,
    skinning_distortion_metrics,
    summarize_character_run,
    topology_preservation_metrics,
)


DEFAULT_ABLATIONS: list[dict[str, Any]] = [
    {"name": "geometric_T008_k4", "use_vit": False, "temperature": 0.08, "k_bones": 4, "command": "walk forward"},
    {"name": "vit_T008_k4", "use_vit": True, "temperature": 0.08, "k_bones": 4, "command": "walk forward"},
    {"name": "vit_T012_k4", "use_vit": True, "temperature": 0.12, "k_bones": 4, "command": "walk forward"},
    {"name": "vit_T020_k4", "use_vit": True, "temperature": 0.20, "k_bones": 4, "command": "walk forward"},
    {"name": "vit_T008_k2", "use_vit": True, "temperature": 0.08, "k_bones": 2, "command": "walk forward"},
    {"name": "vit_T008_k6", "use_vit": True, "temperature": 0.08, "k_bones": 6, "command": "walk forward"},
    {"name": "vit_T008_k4_wave", "use_vit": True, "temperature": 0.08, "k_bones": 4, "command": "wave hello"},
]


def run_character_condition(
    glb_path: str | Path,
    *,
    use_vit: bool,
    temperature: float,
    k_bones: int,
    command: str,
    frames: int = 8,
) -> dict[str, Any]:
    glb_path = Path(glb_path)
    alignment = align_glb_to_standard_skeleton(glb_path, use_vit=use_vit)
    bones = alignment.standard_bones
    vertices = alignment.vertices
    segments = segment_mesh_vertices(vertices, method="front_view")
    bone_indices, weights = compute_vertex_skin_weights(
        vertices,
        bones,
        segments,
        k_bones=k_bones,
        temperature=temperature,
    )
    align_metrics = character_alignment_metrics(
        vertices,
        alignment.triangles,
        bones,
        prediction_source=alignment.prediction.source,
        proportion_scales=alignment.proportion_scales,
    )
    topo = topology_preservation_metrics(bones)
    skin = skinning_distortion_metrics(
        vertices,
        bones,
        bone_indices,
        weights,
        command=command,
        frames=frames,
    )
    report = {
        "glb": str(glb_path.resolve()),
        "use_vit": use_vit,
        "temperature": temperature,
        "k_bones": k_bones,
        "command": command,
        "frames": frames,
        "alignment": align_metrics,
        "topology": topo,
        "skinning": skin,
        "summary": {},
    }
    report["summary"] = summarize_character_run(report)
    return report


def run_character_parameter_study(
    glb_path: str | Path,
    output_dir: str | Path,
    *,
    ablations: list[dict[str, Any]] | None = None,
    frames: int = 8,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations = ablations or DEFAULT_ABLATIONS
    rows: list[dict[str, Any]] = []
    for cfg in ablations:
        name = cfg["name"]
        print(f"[character-study] {name} ...", flush=True)
        report = run_character_condition(
            glb_path,
            use_vit=bool(cfg["use_vit"]),
            temperature=float(cfg["temperature"]),
            k_bones=int(cfg["k_bones"]),
            command=str(cfg["command"]),
            frames=frames,
        )
        report["name"] = name
        (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        row = {"name": name, **report["summary"], "use_vit": report["use_vit"], "temperature": report["temperature"], "k_bones": report["k_bones"], "command": report["command"]}
        rows.append(row)

    table_path = output_dir / "character_study_table.csv"
    _write_csv(table_path, rows)
    summary = {
        "glb": str(Path(glb_path).resolve()),
        "n_conditions": len(rows),
        "table_csv": str(table_path.resolve()),
        "rows": rows,
        "best_by_joint_error": min(rows, key=lambda r: r["mean_joint_error"]),
        "best_by_volume_collapse": max(rows, key=lambda r: r["volume_collapse_ratio"]),
    }
    (output_dir / "character_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
