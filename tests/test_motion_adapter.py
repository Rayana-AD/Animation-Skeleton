"""Tests for motion adapters and language smoke path."""

import numpy as np

from worldmix.character.motion_adapter import (
    list_action_templates,
    select_action_template,
    text_command_to_motion,
    write_motion_json,
)
from worldmix.character.standard_skeleton_gaussians import load_motion_json


def test_text_command_to_motion_and_json(tmp_path):
    sequence = text_command_to_motion("walk forward", frames=6)
    assert len(sequence) == 6
    assert "left_upper_leg" in sequence[1]
    assert "left_upper_arm" in sequence[0]  # natural standing start pose
    path = tmp_path / "motion.json"
    write_motion_json(sequence, path, source="test")
    loaded = load_motion_json(path)
    assert len(loaded) == 6
    assert loaded[1]["left_upper_leg"].shape == (3, 3)


def test_wave_command_moves_arm():
    sequence = text_command_to_motion("wave hand", frames=4)
    assert "right_upper_arm" in sequence[0]
    assert np.linalg.det(sequence[0]["right_upper_arm"]) > 0.9


def test_action_template_selection():
    assert "walk" in list_action_templates()
    assert select_action_template("请让人物自然走路").name == "walk"
    assert select_action_template("wave hand").name == "wave"
