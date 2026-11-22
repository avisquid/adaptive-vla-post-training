"""Train pi0.5 on the RSM cooking data with sunshine's h50/canonical recipe, with discrete state input.

Run from VLA-Adaptation with the pinned upstream OpenPI environment (see README.md):

    pi05_training/openpi/.venv/bin/python pi05_training/train.py --exp-name h50_canonical_dstate
    JAX_PLATFORMS=cpu pi05_training/openpi/.venv/bin/python pi05_training/train.py --exp-name x --preflight
"""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def make_config(args):
    from openpi.models import pi0_config
    from openpi.training import config as _config
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders

    import cooking

    return _config.TrainConfig(
        name="pi05_cooking_h50_canonical_dstate",
        exp_name=args.exp_name,
        project_name="pi05-cooking",
        # The only intended change from sunshine's run: the state enters as discretized
        # prompt tokens ("Task: ..., State: 12 200 ...;\nAction: "), which needs the
        # state quantiles in norm_stats.json. With False, pi0.5 never sees the state.
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True),
        data=cooking.CookingData(
            repo_id=cooking.REPO_ID,
            assets=_config.AssetsConfig(assets_dir=str(HERE / "assets"), asset_id=cooking.ASSET_ID),
            base_config=_config.DataConfig(prompt_from_task=True),  # Canonical instruction per task.
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=args.lr, decay_steps=args.steps, decay_lr=args.lr / 10),
        optimizer=cooking.SplitLRAdamW(clip_gradient_norm=1.0, encoder_lr_scale=0.2),
        ema_decay=0.999,
        seed=0,
        batch_size=args.batch_size,
        num_workers=16,
        num_train_steps=args.steps,
        log_interval=50,
        save_interval=250,
        keep_period=1_000,
        fsdp_devices=args.fsdp_devices,
        assets_base_dir=str(HERE / "assets"),
        checkpoint_base_dir=str(args.checkpoint_dir),
        wandb_enabled=not args.no_wandb,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def install_hooks():
    """Route OpenPI's dataset creation and image augmentation through cooking.py."""
    from openpi.models import model as _model
    from openpi.training import data_loader as _data_loader

    import cooking

    _data_loader.create_torch_dataset = cooking.training_dataset(_data_loader.create_torch_dataset)
    _model.preprocess_observation = cooking.rise_real_preprocess(_model.preprocess_observation)


def preflight(config):
    """Check data, prompts, discrete state and optimizer groups on CPU without training."""
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.models import tokenizer as _tokenizer
    from openpi.training import data_loader as _data_loader

    import cooking

    horizon = config.model.action_horizon
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.norm_stats is not None and data_config.use_quantile_norm
    assert data_config.norm_stats["state"].q01 is not None, "discrete state needs state quantiles"
    raw = _data_loader.create_torch_dataset(data_config, horizon, config.model)
    print("TRAIN_FRAMES", len(raw), flush=True)

    # The last frame of a training segment: its whole chunk must repeat that frame's action.
    import pandas as pd

    frames = pd.concat([pd.read_parquet(p, columns=["index", "annotation.language_remaining"])
                        for p in sorted((cooking.dataset_root(config.data.repo_id) / "data").glob("*/*.parquet"))])
    last = set(frames.loc[frames["annotation.language_remaining"].map(lambda r: int(np.ravel(r)[0])) == 1, "index"])
    position = next(p for p, index in enumerate(raw.indices) if index in last)
    action = raw[position]["action"].numpy()
    assert np.allclose(action, action[0]), "chunk crosses an annotated segment boundary"
    print("SEGMENT_BOUNDARY_OK frame", raw.indices[position], flush=True)

    dataset = _data_loader.transform_dataset(raw, data_config)
    tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
    prompts = set()
    for i in np.random.default_rng(0).choice(len(dataset), 8, replace=False):
        sample = dataset[int(i)]
        assert sample["actions"].shape == (horizon, config.model.action_dim)
        assert np.isfinite(sample["actions"]).all() and np.isfinite(sample["state"]).all()
        mask = sample["tokenized_prompt_mask"]
        assert not mask.all(), "prompt truncated at max_token_len"
        text = tokenizer._tokenizer.decode(sample["tokenized_prompt"][mask].tolist())  # noqa: SLF001
        assert "State: " in text, text
        prompts.add(text.split(", State:")[0])
    print("PROMPT_EXAMPLE", repr(text), flush=True)
    print("TASKS_SEEN", sorted(prompts), flush=True)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_workers=2, num_batches=1)
    batch = next(iter(loader))
    assert batch["actions"].shape == (4, horizon, config.model.action_dim)
    print("MULTIWORKER_BATCH_OK", batch["actions"].shape, batch["tokenized_prompt"].shape, flush=True)
    del loader

    # Optimizer groups on the abstract model (no weights are allocated).
    state = nnx.state(nnx.eval_shape(lambda: config.model.create(jax.random.key(0))))
    params = state.filter(config.trainable_filter)
    counts = {"encoder": 0, "rest": 0}
    for group, leaf in zip(jax.tree.leaves(config.optimizer.labels(params)), jax.tree.leaves(params), strict=True):
        counts[group] += leaf.size
    tx = config.optimizer.create(config.lr_schedule.create())
    jax.eval_shape(tx.init, params)
    print("OPTIMIZER_GROUPS", {k: f"{v / 1e6:.0f}M" for k, v in counts.items()}, flush=True)
    assert counts["encoder"] > 0 and counts["rest"] > 0
    print("PREFLIGHT_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch, divisible by the GPU count")
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate (decays to lr/10)")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check data and config on CPU, then exit")
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(HERE))  # cooking.py, also for spawned data-loader workers.

    install_hooks()
    config = make_config(args)
    if args.preflight:
        preflight(config)
        return

    import importlib.util

    import openpi

    script = Path(openpi.__file__).resolve().parents[2] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    train.main(config)


if __name__ == "__main__":
    main()
