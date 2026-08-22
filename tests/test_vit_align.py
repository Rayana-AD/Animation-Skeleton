"""Tests for ViT alignment scaffold and standard-skeleton GLB overlay."""

import numpy as np

from worldmix.character.standard_skeleton_gaussians import build_standard_humanoid_skeleton, walk_pose
from worldmix.character.vit_align import (
    align_glb_to_standard_skeleton,
    skeleton_joint_targets,
    validate_alignment,
    predict_joints_geometric,
    retarget_skeleton_proportions,
)
from worldmix.character.glb_standard_overlay import verify_motion_path_matches_standard, build_glb_shell_on_standard_skeleton


def test_skeleton_joint_targets_match_bones():
    bones = build_standard_humanoid_skeleton()
    targets = skeleton_joint_targets(bones)
    assert float(targets["head_top"][1]) > float(targets["pelvis"][1])
    assert targets["left_hand"][0] < targets["right_hand"][0]


def test_skeleton_topology_unchanged_after_retarget():
    before = build_standard_humanoid_skeleton()
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(400, 3)) * 0.1
    verts[:, 1] += np.linspace(0.0, 1.0, 400)
    tris = np.array([[0, 1, 2]], dtype=int)
    pred = predict_joints_geometric(verts, tris)
    after, scales = retarget_skeleton_proportions(before, pred.joints)
    assert len(before) == len(after)
    assert [b.name for b in before] == [b.name for b in after]
    assert [b.parent for b in before] == [b.parent for b in after]
    assert scales
    val = validate_alignment(verts, tris, after)
    assert val.hierarchy_preserved


def test_motion_path_matches_standard_for_glb():
    glb = "/Users/rayana/Desktop/01.glb"
    try:
        shell = build_glb_shell_on_standard_skeleton(glb, use_vit=False)
    except FileNotFoundError:
        return
    seq = [walk_pose(0, 4)]
    check = verify_motion_path_matches_standard(shell, seq)
    assert check["same_pipeline"]
    assert check["topology_preserved"]
    assert check["bone_count"] == len(build_standard_humanoid_skeleton())
