"""Pure CPU summary and CLI guard checks; no profiling or model execution."""
import pytest

from scripts import rt_nextlat_a5_fuzzy_batch_profile as profiler


def args(*extra):
    return profiler.parser().parse_args(["--a5-data", "a5", "--fuzzy-data", "fuzzy",
                                        "--output", "fresh-profile", *extra])


def test_summary_uses_aggregate_work_over_time_and_task_lengths():
    result = profiler.summarize_durations([1.0, 3.0], batch_per_task=128)
    assert result["mean_seconds"] == result["median_seconds"] == 2.0
    assert result["p10_seconds"] == pytest.approx(1.2)
    assert result["p90_seconds"] == pytest.approx(2.8)
    assert result["timed_updates"] == 2
    assert result["timed_seconds"] == 4.0
    # 512 total examples / 4 sec. Averaging reciprocal step times is incorrect.
    assert result["total_examples_per_second"] == 128.0
    assert result["per_task"]["a5"]["examples_per_second"] == 64.0
    assert result["per_task"]["a5"]["tokens_per_second"] == 768.0
    assert result["per_task"]["fuzzy"]["tokens_per_second"] == 25600.0
    assert result["total_tokens_per_second"] == 26368.0


@pytest.mark.parametrize("durations", [[], [0.0], [-1.0], [float("nan")], [float("inf")], [[1.0]]])
def test_invalid_timing_samples_are_rejected(durations):
    with pytest.raises(ValueError, match="Timing samples"):
        profiler.summarize_durations(durations, batch_per_task=128)


@pytest.mark.parametrize("extra", [
    ("--batches", "128", "128"), ("--batches", "0"),
    ("--batches", "-1"), ("--warmup-updates", "0"), ("--timed-updates", "0"),
])
def test_candidate_and_phase_guards(extra):
    with pytest.raises(ValueError):
        profiler.validate_arguments(args(*extra))


def test_defaults_keep_bounded_online_throughput_scope():
    defaults = args()
    profiler.validate_arguments(defaults)
    assert defaults.batches == [128, 256, 512, 1024]
    assert defaults.warmup_updates == 10
    assert defaults.timed_updates == 30
    assert defaults.wandb_project == "rt-nextlat-fuzzy-a5"
    assert not hasattr(defaults, "resume")
    assert not hasattr(defaults, "wandb_mode")
    assert not hasattr(defaults, "learning_rate")
    override = args("--batches", "256", "512", "--warmup-updates", "3", "--timed-updates", "7")
    profiler.validate_arguments(override)
    assert override.batches == [256, 512]
    assert (override.warmup_updates, override.timed_updates) == (3, 7)
