# Repository layout

- This branch only trains pi0.5. `VLA-Adaptation/pi05_training/` is the whole project;
  run its commands from `VLA-Adaptation/`. Robot control, data collection and
  deployment live on other branches.

# Training

- OpenPI is upstream `215abfb217dbac7d5f1273282331b9b1866c0479`, unmodified, installed
  by `pi05_training/setup.sh` into its own uv environment. Do not patch OpenPI: put
  task code in `pi05_training/cooking.py` and route OpenPI through it from `train.py`
  using public hooks (`data_loader.create_torch_dataset`,
  `model.preprocess_observation`, a custom `OptimizerConfig`).
- The training data is `xinyunsunshine/rsm_cooking_v5_language` at the revision in
  `cooking.py`: the episodes of `jmarangola/rsm_cooking_v5` plus task labels, the SFT
  split (train on the 223 training episodes only) and segment boundaries.
- Keep the recipe table in `pi05_training/README.md` in step with `train.py`, and mark
  which values are recorded by sunshine's run and which are inferred.
- `discrete_state_input=True` is intentional. pi0.5 has no continuous state input, so
  with `False` the model receives no state at all.
- Run `JAX_PLATFORMS=cpu <openpi python> pi05_training/train.py --exp-name check --preflight`
  after changing data, transforms, augmentation or the optimizer.
- Keep checkpoints, datasets and the OpenPI checkout out of the repository.
