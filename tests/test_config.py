"""Regression tests for the portable configuration seed contract."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu import load_config


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "paper" / "six_channel_160_40.yaml"


def test_named_seed_propagates_to_global_components() -> None:
    config = load_config(
        CONFIG_PATH,
        base_dir=REPOSITORY_ROOT,
        device="cpu",
        seed=123,
    )

    assert config.seed == 123
    assert config.sampling.seed == 123
    assert config.classifiers.random_forest.random_state == 123


def test_dotted_component_seed_overrides_win() -> None:
    config = load_config(
        CONFIG_PATH,
        base_dir=REPOSITORY_ROOT,
        device="cpu",
        seed=123,
        overrides={
            "sampling.seed": 7,
            "classifiers.random_forest.random_state": 11,
        },
    )

    assert config.seed == 123
    assert config.sampling.seed == 7
    assert config.classifiers.random_forest.random_state == 11


def test_with_overrides_named_seed_propagates_to_global_components() -> None:
    base = load_config(CONFIG_PATH, base_dir=REPOSITORY_ROOT, device="cpu")
    config = base.with_overrides(seed=9)

    assert config.seed == 9
    assert config.sampling.seed == 9
    assert config.classifiers.random_forest.random_state == 9


def test_with_overrides_dotted_component_seed_overrides_win() -> None:
    base = load_config(CONFIG_PATH, base_dir=REPOSITORY_ROOT, device="cpu")
    config = base.with_overrides(
        seed=9,
        overrides={
            "sampling.seed": 13,
            "classifiers.random_forest.random_state": 15,
        },
    )

    assert config.seed == 9
    assert config.sampling.seed == 13
    assert config.classifiers.random_forest.random_state == 15
