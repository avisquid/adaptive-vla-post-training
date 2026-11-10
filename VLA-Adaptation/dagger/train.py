"""Fine-tune a pi0.5 checkpoint on the correction frames of one rollout dataset (DAgger).

Run from VLA-Adaptation in the rsm environment, on a GPU that fits a full pi0.5
fine-tune (the stock checkpoints were trained on one 80 GB H100):

    python -m dagger.train ~/lrds/DAgger_part1 --exp-name kitchen_dagger_1

The model, transforms and image augmentation come from the checkpoint's OpenPI config,
and normalization reuses the checkpoint's own statistics, so the result rolls out with
the same rollout.config_name. There is no validation split.
"""
import argparse
import dataclasses
import importlib.util
import os
from pathlib import Path

from dagger.frames import correction_indices

# The git-ignored checkpoints/ folder, which also holds the kitchen SFT checkpoint.
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Rollout dataset directory, <collect.dataset_root>/<repo_id>")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINTS / "pi05-kitchen-sft-v5/h50/canonical/checkpoints/22000")
    parser.add_argument("--config-name", default="pi05_cooking_stock_ft50_s0", help="The checkpoint's OpenPI config")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS / "dagger")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(dataset.parent)  # LeRobot reads it when imported.
    import torch.utils.data
    import openpi
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as _weight_loaders

    base = _config.get_config(args.config_name) #! What is config?
    # The statistics the config reads at inference, where the deployment links them.
    asset_id = base.data.assets.asset_id or base.data.repo_id
    if not (checkpoint / "assets" / asset_id / "norm_stats.json").is_file():
        raise SystemExit(f"{checkpoint} has no assets/{asset_id}/norm_stats.json; link it as docs/bc_setup.md describes.")
    config = dataclasses.replace(
        base,
        exp_name=args.exp_name,
        project_name="rsm_dagger",
        data=dataclasses.replace(
            base.data, repo_id=dataset.name, split="all", knn_pairs=None, adv_path=None,
            assets=_config.AssetsConfig(assets_dir=str(checkpoint / "assets"),
                                        asset_id=asset_id)),
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(checkpoint / "params")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(100, args.steps // 10), peak_lr=args.lr, decay_steps=args.steps, decay_lr=args.lr / 10),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        ema_decay=None,
        eval_interval=0,
        save_interval=500,
        keep_period=None,
        checkpoint_base_dir=str(args.out.expanduser()),
        wandb_enabled=not args.no_wandb,
        overwrite=args.overwrite,
    )

    indices = correction_indices(dataset, config.model.action_horizon)
    if len(indices) < config.batch_size:
        raise SystemExit(f"{dataset.name} has {len(indices)} correction chunks of "
                         f"{config.model.action_horizon} frames; at least one batch ({config.batch_size}) is needed.")
    print(f"DAgger: {len(indices)} correction chunks from {dataset.name}; writing {config.checkpoint_dir}", flush=True)

    # Train on the correction frames only: OpenPI builds its dataset through this function.
    create_dataset = _data_loader.create_torch_dataset
    _data_loader.create_torch_dataset = lambda *a, **k: torch.utils.data.Subset(create_dataset(*a, **k), indices)

    script = Path(openpi.__file__).resolve().parents[2] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    train.main(config)


if __name__ == "__main__":
    main()
