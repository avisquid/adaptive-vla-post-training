"""Fine-tune the supplied OpenPI cooking checkpoint on pinned DAgger_1."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

def main():
    HERE = Path(__file__).resolve().parent
    SOURCE = Path(os.environ['OPENPI_ROOT']).expanduser().resolve()
    BASE = HERE / 'base/h50/canonical/checkpoints/22000'
    DATASET = Path(os.environ.get('WEB_TRAIN_DATASET', str(HERE / 'dataset')))
    OUTPUT = Path(os.environ.get('WEB_TRAIN_OUTPUT', str(HERE / 'export')))
    HOME = HERE / 'lerobot-home'
    (HOME / 'avisquid').mkdir(parents=True, exist_ok=True)
    link = HOME / 'avisquid/DAgger_1'
    if not link.exists():
        link.symlink_to(DATASET, target_is_directory=True)
    os.environ['HF_LEROBOT_HOME'] = str(HOME)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('OPENPI_DATA_HOME', str(HERE.parent / 'openpi-assets'))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.path.insert(0, str(SOURCE / 'src'))

    import numpy as np
    from openpi.training import config as configs, data_loader, optimizer, weight_loaders

    rng = np.random.default_rng(0)
    valid = set(int(x) for x in rng.permutation(32)[:4])
    split = HERE / 'split.json'
    split.write_text(json.dumps({'episodes': {str(i): {'split': 'valid' if i in valid else 'train'} for i in range(32)}}, indent=2))
    parent = configs.get_config('pi05_cooking_stock_ft50_s0')
    data = dataclasses.replace(parent.data, repo_id='avisquid/DAgger_1', split_json=str(split),
        assets=configs.AssetsConfig(assets_dir=str(BASE / 'assets'), asset_id='local/rsm_cooking_v5_language'))
    cfg = dataclasses.replace(parent, name='pi05_dagger1_h50', exp_name='from_22000_s0', seed=0,
        data=data, num_train_steps=1000, batch_size=32, num_workers=8, fsdp_devices=4,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-5, decay_steps=1000, decay_lr=1e-6),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE / 'params')),
        checkpoint_base_dir=str(HERE / 'training-checkpoints'), assets_base_dir=str(HERE / 'assets'),
        eval_interval=250, num_eval_batches=4, log_interval=10, save_interval=100, keep_period=None,
        wandb_enabled=False, resume=(HERE / 'training-checkpoints/pi05_dagger1_h50/from_22000_s0').exists())
    if '--preflight' in sys.argv:
        dc = data.create(cfg.assets_dirs, cfg.model)
        assert dc.norm_stats is not None
        ds = data_loader.create_torch_dataset(dc, 50, cfg.model)
        transformed = data_loader.transform_dataset(ds, dc)
        sample = transformed[0]
        print('TRAIN_FRAMES', len(ds), 'VALID_EPISODES', sorted(valid), flush=True)
        print('ACTION_SHAPE', sample['actions'].shape, 'STATE_SHAPE', sample['state'].shape, flush=True)
        assert sample['actions'].shape == (50, 32)
        assert np.isfinite(sample['actions']).all()
        loader = data_loader.TorchDataLoader(transformed, local_batch_size=4, num_workers=8, num_batches=1)
        batch = next(iter(loader))
        assert batch['actions'].shape == (4, 50, 32)
        print('MULTIWORKER_BATCH_OK', batch['actions'].shape, flush=True)
        del loader
        print('PREFLIGHT_OK', flush=True)
        raise SystemExit(0)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('dagger_openpi_train', SOURCE / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    original_init = train.init_wandb

    def init_wandb(*args, **kwargs):
        original_init(*args, **kwargs)
        original_log = train.wandb.log

        def report(metrics, step=None, **kw):
            original_log(metrics, step=step, **kw)
            record = {'step': int(step or 0), 'total_steps': 1000}
            for key, value in metrics.items():
                try:
                    record[key] = float(value)
                except (ValueError, TypeError):
                    pass
            tmp = OUTPUT / 'progress.tmp'
            tmp.write_text(json.dumps(record))
            tmp.replace(OUTPUT / 'progress.json')

        train.wandb.log = report

    original_loader_init = data_loader.TorchDataLoader.__init__

    def bounded_loader_init(self, *args, **kwargs):
        original_loader_init(self, *args, **kwargs)
        if self._data_loader.num_workers:
            self._data_loader.timeout = 180

    data_loader.TorchDataLoader.__init__ = bounded_loader_init
    train.init_wandb = init_wandb
    print('CONFIG', cfg, flush=True)
    train.main(cfg)
    checkpoints = sorted((p for p in cfg.checkpoint_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    final = checkpoints[-1]
    assert final.name == '999', final
    for item in ['params', 'assets']:
        shutil.copytree(final / item, OUTPUT / item, dirs_exist_ok=True)
    for filename in ['source.json', 'base-source.json', 'split.json', 'audit.json', 'train_dagger.py']:
        shutil.copy2(HERE / filename, OUTPUT / filename)
    shutil.copy2(cfg.checkpoint_dir / 'eval_history.jsonl', OUTPUT / 'eval_history.jsonl')
    (OUTPUT / 'training_config.txt').write_text(repr(cfg))
    (OUTPUT / 'README.md').write_text('''# DAgger_1 pi0.5 fine-tune

    Initialized from xinyunsunshine/pi05-kitchen-sft-v5, h50/canonical/checkpoints/22000.
    1000 updates, batch 32, seed 0, peak LR 1e-5; 28 train episodes, 4 held out.
    All recorded actions in training episodes are supervised, including policy actions.
    OpenPI cooking config pi05_cooking_stock_ft50_s0: action_horizon=50,
    discrete_state_input=False, physical action dimension 10 padded to 32.
    Use global and wrist images; observer camera is not used by this model.
    Preserves the parent normalization in assets/local/rsm_cooking_v5_language.
    This folder contains EMA inference parameters in Orbax format, not optimizer state.
    See source.json and base-source.json for pinned HF revisions; split.json for held-out episodes.
    Offline validation metrics do not establish robot rollout success.
    ''')
    print('FINAL_CHECKPOINT', str(OUTPUT), flush=True)


if __name__ == '__main__':
    main()
