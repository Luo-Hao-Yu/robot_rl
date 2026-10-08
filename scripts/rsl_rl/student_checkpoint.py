"""Initialize a PPO actor from a distillation student's network weights."""

from collections.abc import Mapping
from pathlib import Path

import torch


def initialize_student_actor(runner, checkpoint_path: str | Path) -> int | None:
    """Load only ``student.*`` weights, leaving PPO training state untouched.

    The full distillation checkpoint cannot be resumed as a PPO checkpoint: its
    model and optimizer have different structures. Validate all actor keys and
    shapes before loading so a mismatched observation configuration fails early.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Distillation checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError(f"Invalid distillation checkpoint: {checkpoint_path} has no model_state_dict")

    student_state = {
        name.removeprefix("student."): value
        for name, value in checkpoint["model_state_dict"].items()
        if name.startswith("student.")
    }
    if not student_state:
        raise ValueError(f"Checkpoint contains no student.* weights: {checkpoint_path}")

    actor = runner.alg.policy.actor
    expected_state = actor.state_dict()
    missing = sorted(expected_state.keys() - student_state.keys())
    unexpected = sorted(student_state.keys() - expected_state.keys())
    mismatched = []
    for name in sorted(expected_state.keys() & student_state.keys()):
        value = student_state[name]
        if not isinstance(value, torch.Tensor):
            mismatched.append(f"{name}: checkpoint value is {type(value).__name__}, expected a tensor")
        elif value.shape != expected_state[name].shape:
            mismatched.append(f"{name}: checkpoint {tuple(value.shape)} != actor {tuple(expected_state[name].shape)}")
    if missing or unexpected or mismatched:
        raise ValueError(
            "Distilled student is incompatible with the PPO actor "
            f"(check observation dimensions and network configuration): "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
        )

    actor.load_state_dict(student_state, strict=True)
    return checkpoint.get("iter")
