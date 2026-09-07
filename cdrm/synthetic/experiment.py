"""Shared task dispatch and deterministic paired training stream."""

from cdrm.synthetic.common import concatenate
from cdrm.synthetic.retrieval import (generate_mqar, generate_noisy_recall,
                                     retrieval_baselines, retrieval_oracle, task_spec as retrieval_spec)
from cdrm.synthetic.state_tracking import (generate_state_tracking, interpret_state_tracking,
                                          state_tracking_baselines, task_spec as state_spec)
import numpy as np


def task_spec(task, config):
    return state_spec(config) if task == "state_tracking" else retrieval_spec(task, config)


def generate(task, config, split, seed, count):
    functions = {"mqar": generate_mqar, "noisy_recall": generate_noisy_recall,
                 "state_tracking": generate_state_tracking}
    return functions[task](config, split, seed, count)


def baselines(task, config, batch):
    if task == "state_tracking":
        return state_tracking_baselines(batch, config)
    return retrieval_baselines(batch, task, config)


def oracle(task, config, batch):
    if task != "state_tracking":
        return retrieval_oracle(batch, task, config)
    labels = np.full_like(batch.labels, -100)
    for i, row in enumerate(batch.input_ids):
        labels[i, -1] = interpret_state_tracking(row, config)
    return labels


def generate_training_batch(plan, task, completed_update, split="train"):
    """Batch is a pure function of task, split, seed, and completed-update count.

    Both topologies see identical examples regardless of global RNG consumption
    or checkpoint reload. Conditions are balanced within the global batch.
    """
    if completed_update < 0:
        raise ValueError("completed_update must be nonnegative")
    conditions = plan["tasks"][task]["training_conditions"]
    size = plan["training"]["global_batch"]
    if size % len(conditions):
        raise ValueError("Global batch must divide evenly across training conditions")
    base_seed = plan["seed"] if split == "train" else plan["calibration"]["seed"]
    seed = base_seed * 100_000_000 + completed_update
    return concatenate([generate(task, config, split, seed + index * 10_000_000,
                                 size // len(conditions))
                        for index, config in enumerate(conditions)])
