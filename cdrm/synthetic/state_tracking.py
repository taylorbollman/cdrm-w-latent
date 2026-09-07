"""Ordered finite-state updates; a custom Stage B diagnostic, not a MAD task.

Labels are already aligned with model logits.  Only the terminal ANSWER marker
is scored; the correct answer is never present in the input sequence.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .common import SyntheticBatch


# All operations are bijective and disagree at every source state.  Consequently
# changing ANY one operation changes the final answer, even after later updates.
PERMUTATIONS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5, 0),
    (2, 4, 0, 5, 1, 3),
    (5, 0, 1, 2, 3, 4),
)
HELDOUT_ADJACENT_PAIR = (1, 2)
BOS, INIT, UPDATE, QUERY, ANSWER = range(5)
GENERATOR_REVISION = "ordered-state-v1"


def _configuration(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "sequence_length": 128,
        "num_entities": 3,
        "relevant_updates": 4,
        "distractor_updates": 4,
        "entity_vocab_size": 4,
        "num_states": 6,
        "noise_vocab_size": 8,
        "minimum_query_delay": 0,
        "composition_partition": "train",
        "entity_partition": "train",
    }
    result = {key: config.get(key, value) for key, value in defaults.items()}
    for key in defaults:
        if key.endswith("partition"):
            continue
        if isinstance(result[key], bool) or not isinstance(result[key], int):
            raise ValueError(f"{key} must be an integer")
    if result["num_states"] != 6:
        raise ValueError("The fixed permutation interpreter has exactly six states")
    if not 2 <= result["num_entities"] <= result["entity_vocab_size"]:
        raise ValueError("num_entities must be between two and entity_vocab_size")
    if result["entity_vocab_size"] < 3:
        raise ValueError("At least three known entity symbols are required")
    if result["relevant_updates"] < 1 or result["distractor_updates"] < 0:
        raise ValueError("Positive relevant_updates and nonnegative distractor_updates required")
    if result["noise_vocab_size"] < 1 or result["minimum_query_delay"] < 0:
        raise ValueError("Positive noise vocabulary and nonnegative minimum delay required")
    if result["composition_partition"] not in {"train", "heldout"}:
        raise ValueError("composition_partition must be train or heldout")
    if result["entity_partition"] not in {"train", "heldout"}:
        raise ValueError("entity_partition must be train or heldout")
    if result["composition_partition"] == "heldout" and result["relevant_updates"] < 2:
        raise ValueError("A held-out operation pair requires at least two updates")
    occupied = 1 + 3 * (
        result["num_entities"] + result["relevant_updates"] + result["distractor_updates"]
    ) + 3
    if result["sequence_length"] < occupied + result["minimum_query_delay"]:
        raise ValueError("sequence_length cannot fit the records and requested query delay")
    return result


def task_spec(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _configuration(config)
    entity_start = 5
    state_start = entity_start + cfg["entity_vocab_size"]
    operation_start = state_start + 6
    noise_start = operation_start + len(PERMUTATIONS)
    return {
        "name": "ordered_state_updates",
        "generator_revision": GENERATOR_REVISION,
        "config": cfg,
        "vocab_size": noise_start + cfg["noise_vocab_size"],
        "answer_token_ids": list(range(state_start, state_start + 6)),
        "label_alignment": "already_aligned_with_logits",
        "entity_start": entity_start,
        "state_start": state_start,
        "operation_start": operation_start,
        "noise_start": noise_start,
        "permutations": [list(row) for row in PERMUTATIONS],
        "heldout_adjacent_pair": list(HELDOUT_ADJACENT_PAIR),
        "heldout_query_entity": cfg["entity_vocab_size"] - 1,
    }


def apply_operations(initial_state: int, operations: Sequence[int]) -> int:
    """Exact finite-state oracle, independent of token serialization."""
    if not 0 <= initial_state < 6:
        raise ValueError("initial_state outside six-state alphabet")
    state = initial_state
    for operation in operations:
        if not 0 <= operation < len(PERMUTATIONS):
            raise ValueError("unknown operation")
        state = PERMUTATIONS[operation][state]
    return state


def is_heldout_composition(operations: Sequence[int]) -> bool:
    return any(tuple(pair) == HELDOUT_ADJACENT_PAIR for pair in zip(operations, operations[1:]))


def _stream(seed: int, split: str, index: int, domain: str) -> np.random.Generator:
    message = f"{GENERATOR_REVISION}|{seed}|{len(split)}:{split}|{index}|{domain}"
    digest = hashlib.sha256(message.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _sample_operations(rng: np.random.Generator, count: int, heldout: bool) -> list[int]:
    """Uniform strings conditional on containing/avoiding the held-out pair."""
    @lru_cache(None)
    def ways(remaining: int, previous: int, seen: bool) -> int:
        if remaining == 0:
            return int(seen == heldout)
        total = 0
        for operation in range(len(PERMUTATIONS)):
            next_seen = seen or (previous, operation) == HELDOUT_ADJACENT_PAIR
            if not heldout and next_seen:
                continue
            total += ways(remaining - 1, operation, next_seen)
        return total

    if ways(count, -1, False) == 0:
        raise ValueError("No operation sequence satisfies the requested composition partition")
    result: list[int] = []
    previous, seen = -1, False
    for remaining in range(count, 0, -1):
        weights = []
        for operation in range(len(PERMUTATIONS)):
            next_seen = seen or (previous, operation) == HELDOUT_ADJACENT_PAIR
            weights.append(
                0 if not heldout and next_seen else ways(remaining - 1, operation, next_seen)
            )
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        operation = int(rng.choice(len(PERMUTATIONS), p=probabilities))
        seen = seen or (previous, operation) == HELDOUT_ADJACENT_PAIR
        result.append(operation)
        previous = operation
    return result


def generate_state_tracking(
    config: dict[str, Any], split: str, seed: int, num_examples: int
) -> SyntheticBatch:
    """Generate independent examples without mutating global random state.

    ``split`` names an RNG domain (e.g. train, dev, test).  The two generalization
    conditions are selected explicitly by configuration, not inferred from its
    spelling.  Changing only sequence length preserves the semantic records.
    """
    spec = task_spec(config)
    cfg = spec["config"]
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    if isinstance(num_examples, bool) or not isinstance(num_examples, int) or num_examples < 0:
        raise ValueError("num_examples must be a nonnegative integer")
    input_ids = np.empty((num_examples, cfg["sequence_length"]), dtype=np.int64)
    labels = np.full(input_ids.shape, -100, dtype=np.int64)
    metadata: list[dict[str, Any]] = []
    for index in range(num_examples):
        rng = _stream(seed, split, index, "semantics")
        layout_rng = _stream(seed, split, index, "layout")
        heldout_entity = cfg["entity_vocab_size"] - 1
        target = heldout_entity if cfg["entity_partition"] == "heldout" else int(rng.integers(heldout_entity))
        others = [entity for entity in range(cfg["entity_vocab_size"]) if entity != target]
        selected = [target] + list(map(int, rng.choice(others, cfg["num_entities"] - 1, replace=False)))
        initial_states = {entity: int(rng.integers(6)) for entity in selected}
        entity_schedule = [target] * cfg["relevant_updates"]
        entity_schedule += list(map(int, rng.choice(selected[1:], cfg["distractor_updates"])))
        rng.shuffle(entity_schedule)
        operations = {
            entity: _sample_operations(
                rng,
                entity_schedule.count(entity),
                cfg["composition_partition"] == "heldout" and entity == target,
            )
            for entity in selected
        }
        cursors = dict.fromkeys(selected, 0)
        events: list[dict[str, int]] = []
        for entity in entity_schedule:
            operation = operations[entity][cursors[entity]]
            cursors[entity] += 1
            events.append({"entity": entity, "operation": operation})
        init_order = list(selected)
        rng.shuffle(init_order)
        records = [
            [INIT, spec["entity_start"] + entity, spec["state_start"] + initial_states[entity]]
            for entity in init_order
        ] + [
            [UPDATE, spec["entity_start"] + event["entity"], spec["operation_start"] + event["operation"]]
            for event in events
        ]
        noise_count = cfg["sequence_length"] - 4 - 3 * len(records)
        gap_sizes = layout_rng.multinomial(
            noise_count - cfg["minimum_query_delay"],
            np.full(len(records) + 1, 1 / (len(records) + 1)),
        )
        gap_sizes[-1] += cfg["minimum_query_delay"]
        tokens = [BOS]
        record_positions: list[int] = []
        for record_index in range(len(records) + 1):
            tokens.extend(map(int, layout_rng.integers(
                spec["noise_start"], spec["vocab_size"], size=int(gap_sizes[record_index])
            )))
            if record_index < len(records):
                record_positions.append(len(tokens))
                tokens.extend(records[record_index])
        query_position = len(tokens)
        tokens.extend([QUERY, spec["entity_start"] + target, ANSWER])
        answer_position = len(tokens) - 1
        answer_state = apply_operations(initial_states[target], operations[target])
        input_ids[index] = tokens
        labels[index, answer_position] = spec["state_start"] + answer_state
        relevant_positions = [
            record_positions[cfg["num_entities"] + event_index] + 2
            for event_index, event in enumerate(events) if event["entity"] == target
        ]
        metadata.append({
            "task": spec["name"], "generator_revision": GENERATOR_REVISION,
            "split": split, "seed": seed, "example_index": index,
            "target_entity": target, "entities": selected,
            "initial_states": {str(entity): state for entity, state in initial_states.items()},
            "events": events, "target_operations": operations[target],
            "composition_partition": cfg["composition_partition"],
            "entity_partition": cfg["entity_partition"],
            "answer_state": answer_state, "answer_token_id": spec["state_start"] + answer_state,
            "answer_position": answer_position, "query_position": query_position,
            "num_entities": cfg["num_entities"], "relevant_updates": cfg["relevant_updates"],
            "distractor_updates": cfg["distractor_updates"],
            "delay": query_position - relevant_positions[-1] - 1,
            "first_update_delay": query_position - relevant_positions[0] - 1,
            "relevant_operation_positions": relevant_positions,
            "sequence_length": cfg["sequence_length"],
        })
    return SyntheticBatch(input_ids=input_ids, labels=labels, metadata=metadata)


def interpret_state_tracking(tokens: Sequence[int], config: dict[str, Any]) -> int:
    """Parse records and execute their updates, returning the answer token ID."""
    spec = task_spec(config)
    values = list(map(int, tokens))
    if not values or values[0] != BOS:
        raise ValueError("Missing BOS")
    states: dict[int, int] = {}
    position = 1
    saw_update = False
    while position < len(values):
        token = values[position]
        if spec["noise_start"] <= token < spec["vocab_size"]:
            position += 1
            continue
        if token not in {INIT, UPDATE, QUERY} or position + 2 >= len(values):
            raise ValueError("Malformed or truncated record")
        entity = values[position + 1] - spec["entity_start"]
        if not 0 <= entity < spec["config"]["entity_vocab_size"]:
            raise ValueError("Invalid entity symbol")
        argument = values[position + 2]
        if token == INIT:
            state = argument - spec["state_start"]
            if saw_update or entity in states or not 0 <= state < 6:
                raise ValueError("Invalid or duplicate initialization")
            states[entity] = state
        elif token == UPDATE:
            operation = argument - spec["operation_start"]
            if entity not in states or not 0 <= operation < len(PERMUTATIONS):
                raise ValueError("Update requires initialized entity and known operation")
            states[entity] = PERMUTATIONS[operation][states[entity]]
            saw_update = True
        else:
            if entity not in states or argument != ANSWER or position + 3 != len(values):
                raise ValueError("Exactly one terminal QUERY/entity/ANSWER is required")
            return spec["state_start"] + states[entity]
        position += 3
    raise ValueError("Missing terminal query")


def state_tracking_baselines(batch: SyntheticBatch, config: dict[str, Any]) -> dict[str, Any]:
    """Oracle-informed shortcut checks; none is a learned model baseline.

    Restricted-history baselines are deliberately generous: they retain the
    correct initial state and entity identity, but ignore all but the last one
    or two relevant operations.  Last-assignment retrieves initialization only.
    """
    spec = task_spec(config)
    predictions: dict[str, list[int]] = {
        "last_assignment": [], "last_record": [], "restricted_last_two": [], "oracle": []
    }
    for tokens, row in zip(batch.input_ids, batch.metadata):
        initial = row["initial_states"][str(row["target_entity"])]
        operations = row["target_operations"]
        predictions["last_assignment"].append(spec["state_start"] + initial)
        predictions["last_record"].append(spec["state_start"] + apply_operations(initial, operations[-1:]))
        predictions["restricted_last_two"].append(spec["state_start"] + apply_operations(initial, operations[-2:]))
        predictions["oracle"].append(interpret_state_tracking(tokens, config))
    targets = [int(row[row != -100][0]) for row in batch.labels]
    return {
        "chance_accuracy": 1 / 6,
        "num_examples": len(targets),
        **{
            f"{name}_accuracy": float(np.mean(np.asarray(values) == targets)) if targets else None
            for name, values in predictions.items()
        },
        "restricted_history_definition": "Known initial state plus last one/two relevant operations; all earlier updates ignored",
    }
