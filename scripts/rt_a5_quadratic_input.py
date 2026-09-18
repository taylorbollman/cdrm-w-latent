"""The existing four-layer input bypass with a nonlearned quadratic warmup."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.rt_a5_embedding_injection import build_model as build_fixed_model


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/rt_a5_quadratic_input/base.json"
SCHEDULE = json.loads(CONFIG_PATH.read_text())["schedule"]


def coefficient(update: int) -> float:
    if type(update) is not int:
        raise ValueError("Schedule update must be an integer")
    return .05 * min(max(update, 0) / 50000, 1) ** 2


def set_update(model, update: int) -> float:
    if type(update) is not int or update < 0:
        raise ValueError("Applied schedule update must be a nonnegative integer")
    value = coefficient(update)
    model.backbone.injection_coefficient = value
    model.schedule_update = update
    return value


def build_model(width=512, seed=1234, predictor_seed=1235, device="cpu", *,
                projection_seed=1236, backend="tiled", predictor_hidden_width=None):
    model = build_fixed_model(width, seed, predictor_seed, device,
        projection_seed=projection_seed, coefficient=0., variant="input",
        backend=backend, predictor_hidden_width=predictor_hidden_width)
    original = copy.deepcopy(model.nextlat_initialization)
    config = copy.deepcopy(model.experiment_config)
    config.update(json.loads(CONFIG_PATH.read_text()))
    config.pop("coefficient")
    model.experiment_config = config
    model.nextlat_initialization = {
        **original, "schema": "rt-a5-quadratic-input-initialization-v1",
        "reference_input_initialization": original,
        "schedule": copy.deepcopy(SCHEDULE), "initial_coefficient": 0.,
        "experiment_config": copy.deepcopy(config),
        "rule": config["initialization_pairing"],
    }
    set_update(model, 0)
    return model
