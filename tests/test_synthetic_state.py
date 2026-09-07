"""CPU generator checks: oracle correctness is NUM, never learned SYN evidence."""

from collections import Counter

import numpy as np
import pytest

from cdrm.synthetic.state_tracking import (
    ANSWER,
    INIT,
    PERMUTATIONS,
    QUERY,
    UPDATE,
    apply_operations,
    generate_state_tracking,
    interpret_state_tracking,
    is_heldout_composition,
    state_tracking_baselines,
    task_spec,
)


@pytest.mark.parametrize("length", [128, 256, 512])
@pytest.mark.parametrize("composition,entity", [("train", "train"), ("heldout", "train"), ("train", "heldout")])
def test_oracle_labels_masks_and_serialization(length, composition, entity):
    config = {"sequence_length": length, "composition_partition": composition,
              "entity_partition": entity, "minimum_query_delay": 12}
    spec = task_spec(config)
    batch = generate_state_tracking(config, "test", 813, 32)
    assert batch.input_ids.dtype == batch.labels.dtype == np.int64
    assert batch.input_ids.shape == batch.labels.shape == (32, length)
    assert np.all((batch.labels != -100).sum(axis=1) == 1)
    assert np.all(batch.labels[:, :-1] == -100)
    assert np.all(batch.input_ids[:, -1] == ANSWER)
    assert np.all(batch.input_ids[:, -3] == QUERY)
    assert batch.input_ids.min() >= 0 and batch.input_ids.max() < spec["vocab_size"]
    for tokens, labels, row in zip(batch.input_ids, batch.labels, batch.metadata):
        # An independently written metadata interpreter checks the token parser.
        states = {int(key): value for key, value in row["initial_states"].items()}
        for event in row["events"]:
            states[event["entity"]] = spec["permutations"][event["operation"]][states[event["entity"]]]
        expected = spec["state_start"] + states[row["target_entity"]]
        assert expected == interpret_state_tracking(tokens, config) == labels[-1]
        assert len(row["target_operations"]) == 4 and len(row["events"]) == 8
        assert row["delay"] >= 12
        assert sum(tokens == INIT) == 3 and sum(tokens == UPDATE) == 8
        # State tokens occur only as initial states, never as update answers.
        state_positions = np.flatnonzero(np.isin(tokens, spec["answer_token_ids"]))
        assert len(state_positions) == 3
        assert all(tokens[position - 2] == INIT for position in state_positions)


def test_noncommuting_operations_and_counterfactual_earlier_updates():
    for operation in PERMUTATIONS:
        assert sorted(operation) == list(range(6))
    assert any(apply_operations(state, [0, 1]) != apply_operations(state, [1, 0]) for state in range(6))
    for state in range(6):
        assert len({operation[state] for operation in PERMUTATIONS}) == 3
    config = {"relevant_updates": 8}
    spec = task_spec(config)
    batch = generate_state_tracking(config, "counterfactual", 7, 32)
    for tokens, row in zip(batch.input_ids, batch.metadata):
        answer = interpret_state_tracking(tokens, config)
        for position in row["relevant_operation_positions"]:
            for replacement in range(3):
                if replacement == tokens[position] - spec["operation_start"]:
                    continue
                changed = tokens.copy()
                changed[position] = spec["operation_start"] + replacement
                assert interpret_state_tracking(changed, config) != answer
        # A changed distractor operation must not affect this target's state.
        distractor_positions = [position + 2 for position in np.flatnonzero(tokens == UPDATE)
                               if tokens[position + 1] != spec["entity_start"] + row["target_entity"]]
        for position in distractor_positions:
            changed = tokens.copy()
            changed[position] = spec["operation_start"] + ((tokens[position] - spec["operation_start"] + 1) % 3)
            assert interpret_state_tracking(changed, config) == answer


def test_determinism_split_independence_and_fixed_event_delay_extension():
    base = generate_state_tracking({}, "dev", 4, 48)
    repeated = generate_state_tracking({}, "dev", 4, 48)
    assert base.sha256 == repeated.sha256
    assert base.metadata == repeated.metadata
    assert base.sha256 != generate_state_tracking({}, "test", 4, 48).sha256
    assert base.sha256 != generate_state_tracking({}, "dev", 5, 48).sha256
    for length in (256, 512):
        longer = generate_state_tracking({"sequence_length": length}, "dev", 4, 48)
        assert np.array_equal(base.labels[:, -1], longer.labels[:, -1])
        for first, second in zip(base.metadata, longer.metadata):
            for key in ("target_entity", "entities", "initial_states", "events", "target_operations", "answer_state"):
                assert first[key] == second[key]
        assert np.mean([row["first_update_delay"] for row in longer.metadata]) > np.mean(
            [row["first_update_delay"] for row in base.metadata])
    dev_strings = {tuple(row) for row in base.input_ids}
    test_strings = {tuple(row) for row in generate_state_tracking({}, "test", 4, 48).input_ids}
    assert not dev_strings & test_strings


def test_composition_and_role_partitions_use_known_symbols():
    trained = generate_state_tracking({}, "train", 20, 512)
    composition = generate_state_tracking({"composition_partition": "heldout"}, "test", 20, 128)
    role = generate_state_tracking({"entity_partition": "heldout"}, "test", 20, 128)
    all_entities, all_operations, train_targets = set(), set(), set()
    for row in trained.metadata:
        all_entities.update(row["entities"])
        train_targets.add(row["target_entity"])
        for entity in row["entities"]:
            operations = [event["operation"] for event in row["events"] if event["entity"] == entity]
            all_operations.update(operations)
            assert not is_heldout_composition(operations)
    assert all_entities == {0, 1, 2, 3}
    assert all_operations == {0, 1, 2}
    assert train_targets == {0, 1, 2}
    assert all(is_heldout_composition(row["target_operations"]) for row in composition.metadata)
    assert all(row["target_entity"] == 3 for row in role.metadata)
    assert all(not is_heldout_composition(row["target_operations"]) for row in role.metadata)


def test_answer_balance_and_retrieval_shortcuts_are_insufficient():
    for config in ({}, {"composition_partition": "heldout"}, {"entity_partition": "heldout"}):
        batch = generate_state_tracking(config, "audit", 77, 2048)
        baselines = state_tracking_baselines(batch, config)
        assert baselines["oracle_accuracy"] == 1.0
        assert baselines["chance_accuracy"] == 1 / 6
        for name in ("last_assignment_accuracy", "last_record_accuracy", "restricted_last_two_accuracy"):
            assert baselines[name] < 0.55
        counts = Counter(row["answer_state"] for row in batch.metadata)
        assert len(counts) == 6
        assert all(abs(count / 2048 - 1 / 6) < 0.035 for count in counts.values())
        # Condition on the last operation as well: it must not identify the answer.
        for operation in range(3):
            answers = [row["answer_state"] for row in batch.metadata if row["target_operations"][-1] == operation]
            assert max(Counter(answers).values()) / len(answers) < 0.25


@pytest.mark.parametrize("config", [
    {"sequence_length": 16}, {"num_entities": 1}, {"num_entities": 5},
    {"num_states": 8}, {"relevant_updates": 0}, {"distractor_updates": -1},
    {"composition_partition": "invalid"}, {"entity_partition": "invalid"},
    {"relevant_updates": 1, "composition_partition": "heldout"},
    {"minimum_query_delay": 128}, {"sequence_length": True},
])
def test_invalid_configurations(config):
    with pytest.raises(ValueError):
        generate_state_tracking(config, "dev", 1, 1)


def test_interpreter_rejects_malformed_records_and_missing_query():
    config = {}
    spec = task_spec(config)
    tokens = generate_state_tracking(config, "test", 1, 1).input_ids[0]
    with pytest.raises(ValueError, match="terminal"):
        interpret_state_tracking(list(tokens) + [spec["noise_start"]], config)
    with pytest.raises(ValueError):
        interpret_state_tracking(tokens[:-3], config)
    changed = tokens.copy()
    position = np.flatnonzero(changed == UPDATE)[0]
    changed[position + 2] = spec["state_start"]
    with pytest.raises(ValueError, match="operation"):
        interpret_state_tracking(changed, config)


def test_empty_batch_is_well_formed():
    batch = generate_state_tracking({}, "dev", 8, 0)
    assert batch.input_ids.shape == (0, 128)
    assert state_tracking_baselines(batch, {})["oracle_accuracy"] is None
