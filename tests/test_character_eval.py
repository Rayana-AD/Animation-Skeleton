"""Skeleton-pack tests (character alignment metrics only)."""

from __future__ import annotations

import numpy as np

from worldmix.character.standard_skeleton_gaussians import build_standard_humanoid_skeleton
from worldmix.character.vit_align import predict_joints_geometric, retarget_skeleton_proportions
from worldmix.eval.metrics import character_alignment_metrics, topology_preservation_metrics


def test_topology_preservation_on_template():
    bones = build_standard_humanoid_skeleton()
    assert topology_preservation_metrics(bones)["topology_preserved"]


def test_character_alignment_metrics_synthetic():
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(400, 3)) * 0.1
    verts[:, 1] += np.linspace(0.0, 1.0, 400)
    tris = np.array([[0, 1, 2]], dtype=int)
    template = build_standard_humanoid_skeleton()
    pred = predict_joints_geometric(verts, tris)
    bones, scales = retarget_skeleton_proportions(template, pred.joints)
    m = character_alignment_metrics(verts, tris, bones, proportion_scales=scales)
    assert m["bone_count"] == 53
    assert m["hierarchy_preserved"]
