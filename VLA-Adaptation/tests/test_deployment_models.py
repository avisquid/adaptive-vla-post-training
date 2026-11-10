"""Model-interface regressions without checkpoint weights or robot hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from mock_rollout import mock_rollout
from real.deployment.config import load_rollout, worker_config
from real.deployment import models


def observation(t=1):
    state = np.array([.4, 0, .2, 1, 0, 0, 0, 1, 0, .04], np.float32)
    images = {name: np.zeros((12, 16, 3), np.uint8) for name in ("global", "wrist", "visualization")}
    return {"images": images, "state": state, "t": t, "history": [], "task": "fixture"}


def mock_config(**overrides):
    return worker_config(mock_rollout(**{"execute_k": None, **overrides}), simulate=True)


def openpi_config(checkpoint, **overrides):
    return worker_config(load_rollout("pi05_stock", checkpoint=str(checkpoint), device="cpu", **overrides),
                         simulate=False)


def test_mock_policy_returns_one_chunk():
    planner = models.build_planner(mock_config())
    plan = planner.plan(observation(), 21)
    assert plan["chosen_actions"].shape == (8, 10)
    assert plan["chosen_actions"].dtype == np.float32
    assert plan["timings"]["planning_s"] >= 0
    assert planner.metadata()["mock"] is True
    assert planner.metadata()["inference_hardware"]["proposal"]["type"] == "cpu"
    assert planner.metadata()["inference_hardware"]["proposal"]["model"]


def test_proposal_seed_is_independent_of_prior_calls():
    planner = models.build_planner(mock_config())
    before = planner.plan(observation(), 14)
    # Disturb global RNG and the planner's previous sampling state.
    np.random.seed(1337)
    np.random.uniform(size=100)
    planner.plan(observation(), 103)
    planner.reset()
    after = planner.plan(observation(), 14)
    np.testing.assert_array_equal(before["chosen_actions"], after["chosen_actions"])
    assert before["proposal_seed"] == after["proposal_seed"]


def test_model_inputs_exclude_recording_only_views_and_keep_control_history():
    raw = observation(8)
    raw["history"] = [observation(7)]
    canonical = models._canonical_obs(raw, ["global", "wrist"])
    assert set(canonical["images"]) == {"global", "wrist"}
    assert canonical["history"][0]["t"] == 7
    assert set(raw["images"]) == {"global", "wrist", "visualization"}


def test_namespace_package_dependency_is_in_recorded_source_roots(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "openpi"
    source.mkdir()
    monkeypatch.setitem(sys.modules, "openpi", SimpleNamespace(__file__=None, __path__=[str(source)]))
    assert models._source_roots(False)["openpi_0"] == str(source.resolve())
    assert models._source_roots(True) == {}


@pytest.mark.parametrize("overrides, match", [
    ({"kind": "act"}, "rollout.kind"),
    ({"cameras": ["global", "global"]}, "distinct"),
    ({"cameras": ["wrist"]}, "exactly two cameras"),
    ({"execute_k": 0}, "execute_k"),
    ({"inference": {"environment": {"BAD-NAME": "1"}}}, "environment"),
])
def test_invalid_presets_rejected_before_a_worker_starts(overrides, match):
    with pytest.raises(ValueError, match=match):
        mock_config(**overrides)


def test_mock_requires_simulation_and_openpi_requires_its_config(tmp_path):
    with pytest.raises(ValueError, match="simulate"):
        worker_config(mock_rollout(), simulate=False)
    with pytest.raises(ValueError, match="config_name"):
        openpi_config(tmp_path, config_name=None)
    with pytest.raises(FileNotFoundError, match="checkpoint directory"):
        openpi_config(tmp_path / "missing")


@pytest.mark.parametrize("overrides, match", [({"candidate_horizon": 0}, "positive"), ({"execute_k": 9}, "execute_k")])
def test_horizons_are_checked_against_the_loaded_policy(overrides, match):
    with pytest.raises(ValueError, match=match):
        models.build_planner(mock_config(**overrides))


def test_nonfinite_actions_are_rejected(monkeypatch):
    planner = models.build_planner(mock_config())
    monkeypatch.setattr(planner.proposer, "candidates", lambda *_: np.full((1, 8, 10), np.nan))
    with pytest.raises(ValueError, match="finite absolute"):
        planner.plan(observation(), 1)


def test_learned_default_horizon_comes_from_checkpoint(tmp_path, monkeypatch):
    from real.policies import openpi

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cfg = openpi_config(checkpoint, execute_k=None)
    fake = models._MockProposal(16)
    monkeypatch.setattr(models, "dependency_report", lambda *_: {"missing": [], "python": "fixture"})
    monkeypatch.setattr(models, "_hardware_info", lambda *_: {})
    monkeypatch.setattr(openpi, "OpenpiPolicy", lambda *_, **__: fake)
    planner = models.build_planner(cfg)
    assert planner.metadata()["candidate_horizon"] == 16
    assert planner.plan(observation(), 0)["chosen_actions"].shape == (16, 10)


def test_missing_dependency_diagnostics_do_not_fabricate_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_available", lambda _: False)
    report = models.dependency_report("openpi")
    assert any("Openpi" in message for message in report["missing"])
    with pytest.raises(models.ModelDependencyError, match="pinned training dependencies"):
        models.build_planner(openpi_config(tmp_path))


def test_openpi_does_not_silently_fall_back(monkeypatch):
    from real.policies.openpi import OpenpiPolicy

    policy = OpenpiPolicy.__new__(OpenpiPolicy)
    policy.H, policy.D, policy.batch, policy._batched_ok = 8, 10, 4, True
    policy.rng = np.random.default_rng(0)
    def broken(*_):
        raise RuntimeError("batched path broke")
    monkeypatch.setattr(policy, "_candidates_batched", broken)
    monkeypatch.setattr(policy, "_candidates_serial", lambda *_: pytest.fail("must not fall back"))
    with pytest.raises(RuntimeError, match="batched path broke"):
        policy.candidates({}, 4, 8)


def test_dry_run_worker_waits_without_timeouts():
    assert mock_config()["inference"]["timeout_s"] == 60
    config = worker_config(mock_rollout(), simulate=True, wait_forever=True)
    assert config["inference"]["timeout_s"] is None
    from real.deployment.worker import ModelWorker
    worker = ModelWorker(config)
    try:
        assert worker.timeout_s is None
        worker.reset()
        plan = worker.submit_plan(observation(), 3).result()
        assert plan["prompt"] == "fixture"
    finally:
        worker.close()
