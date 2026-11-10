# DAgger fine-tuning

Fine-tunes a deployed pi0.5 checkpoint on the operator's SpaceMouse corrections from
one rollout dataset (collected with `rollout=<preset>`; see
`docs/single_arm_collection.md`).

- `frames.py` picks the training frames: `intervention=1`, excluding pauses (from the
  episode `handoffs`). A frame is used only if its whole action chunk (the model's
  action horizon: 50 frames, 1 s, for the kitchen SFT checkpoint) is corrections from
  the same episode, so policy actions and paused holds are never targets. Pure
  SpaceMouse episodes (**S**) in a rollout dataset count as corrections throughout.
- `train.py` runs OpenPI's own training loop on those frames, starting from the
  checkpoint's weights and reusing its normalization statistics. No validation split.

Run from `VLA-Adaptation/` in `rsm`, on a GPU that fits a full pi0.5 fine-tune (the
stock checkpoints used one 80 GB H100; a 24 GB card does not fit). This is far smaller
than the kitchen SFT run itself: 8 GPUs, batch 256, 30,000 steps
(`h50/canonical/training_config.json`).

```bash
python -m dagger.train /home/ankile/lrds/DAgger_part1 --exp-name kitchen_dagger_1
```

Defaults: the kitchen SFT checkpoint (`h50/canonical`, step 22000) and its config
`pi05_cooking_stock_ft50_s0`, 2000 steps, batch 32, peak learning rate 1e-5 (cosine),
no EMA. Pass `--checkpoint` and `--config-name` together to start from another
checkpoint, `--no-wandb` to skip logging. Checkpoints go to the git-ignored
`checkpoints/dagger/<config-name>/<exp-name>/<step>/`; roll the last one out with the
same preset:

```bash
python -m real.collect_sft_data task=tabletop rollout=pi05_kitchen_sft_v5 \
  rollout.checkpoint=checkpoints/dagger/pi05_cooking_stock_ft50_s0/kitchen_dagger_1/1999 ...
```

## All-action DAgger and union comparison

The separate [cluster training workflow](../dagger_training/README.md) reproduces
the completed DAgger_1 and DAgger + cooking-language experiments with pinned
Hugging Face inputs, episode holdouts, Slurm submission, and checkpoint export.
It trains on all recorded actions, including policy actions; it does not apply
this directory's intervention-only filter.
