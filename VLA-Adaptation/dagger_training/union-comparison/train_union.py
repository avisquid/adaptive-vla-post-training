"""Matched union fine-tune and reevaluation of the existing DAgger-only policy."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ['OPENPI_ROOT']).expanduser().resolve()
BASE=ROOT.parent/'dagger-1/base/h50/canonical/checkpoints/22000'
DAGGER=Path(os.environ.get('DAGGER_CHECKPOINT', str(ROOT.parent/'dagger-1/export'))).expanduser().resolve()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['union','dagger_eval'],default='union')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    home=ROOT/'lerobot-home'
    for repo,target in [('avisquid/DAgger_1',ROOT.parent/'dagger-1/dataset'),('xinyunsunshine/rsm_cooking_v5_language',ROOT/'language')]:
        link=home/repo;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(target,target_is_directory=True)
    os.environ.update(HF_LEROBOT_HOME=str(home),HF_HUB_OFFLINE='1',OPENPI_DATA_HOME=os.environ.get('OPENPI_DATA_HOME',str(ROOT.parent/'openpi-assets')),
        TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',PYTHONUNBUFFERED='1')
    sys.path.insert(0,str(SOURCE/'src'))
    import numpy as np
    from openpi.training import config as configs, data_loader, optimizer, weight_loaders
    import union_data
    data_loader.create_torch_dataset=union_data.union_dataset
    parent=configs.get_config('pi05_cooking_stock_ft50_s0')
    data=dataclasses.replace(parent.data,repo_id='local/cooking_union',split='all',split_json=None,
        assets=configs.AssetsConfig(assets_dir=str(BASE/'assets'),asset_id='local/rsm_cooking_v5_language'))
    name='pi05_union_h50' if args.mode=='union' else 'pi05_dagger_only_eval'
    ckdir=ROOT/'training-checkpoints'/name/'from_22000_s0'
    cfg=dataclasses.replace(parent,name=name,exp_name='from_22000_s0',seed=0,data=data,
        num_train_steps=1000 if args.mode=='union' else 0,batch_size=32,num_workers=8,fsdp_devices=4,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=1e-5,decay_steps=1000,decay_lr=1e-6),
        weight_loader=weight_loaders.CheckpointWeightLoader(str((BASE if args.mode=='union' else DAGGER)/'params')),
        checkpoint_base_dir=str(ROOT/'training-checkpoints'),assets_base_dir=str(ROOT/'assets'),
        eval_interval=250,num_eval_batches=4,log_interval=10,save_interval=100,keep_period=None,
        wandb_enabled=False,resume=ckdir.exists())
    dc=data.create(cfg.assets_dirs,cfg.model)
    if args.preflight:
        raw=union_data.union_dataset(dc,50,cfg.model)
        assert len(raw)==193506
        for name,ds in zip(['dagger','language'],raw.datasets,strict=True):
            transformed=data_loader.transform_dataset(ds,dc)
            sample=transformed[0]
            assert sample['actions'].shape==(50,32) and np.isfinite(sample['actions']).all()
            loader=data_loader.TorchDataLoader(transformed,local_batch_size=4,num_workers=8,num_batches=1)
            batch=next(iter(loader));assert batch['actions'].shape==(4,50,32)
            print('MULTIWORKER_OK',name,len(ds),flush=True)
            del loader
        # Exercise a real language segment tail, not only the first frame.
        import pyarrow.parquet as pq
        ep=union_data.MANIFEST['splits']['language']['train'][0]
        t=pq.read_table(ROOT/f'language/data/chunk-000/episode_{ep:06d}.parquet')
        remaining=np.asarray(t['annotation.language_remaining'].to_pylist()).reshape(-1)
        idx=int(np.flatnonzero(remaining==1)[0]);item=raw.datasets[1][idx]
        assert np.allclose(item['action'].numpy(),item['action'][0].numpy())
        print('SEGMENT_BOUNDARY_OK',ep,idx,flush=True)
        return
    output=ROOT/'exports'/args.mode;output.mkdir(parents=True,exist_ok=True)
    # Keep upstream immutable; change only diagnostic split names and the language evaluation route.
    text=(SOURCE/'scripts/train.py').read_text()
    if 'make_eval_loader(config, "train",' not in text:
        raise RuntimeError('OpenPI training loop lacks the expected evaluation route; use the documented cooking checkout')
    text=text.replace('"val"','"dagger_val"').replace('"train_eval"','"language_val"')
    text=text.replace('make_eval_loader(config, "train",','make_eval_loader(config, "language_valid",')
    module_path=ROOT/'openpi_train_comparison.py';module_path.write_text(text)
    spec=importlib.util.spec_from_file_location('openpi_train_comparison',module_path)
    train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)
    train.make_eval_loader=union_data.evaluation_loader
    original_init=train.init_wandb
    def init_wandb(*a,**kw):
        original_init(*a,**kw)
        original_log=train.wandb.log
        def log(metrics,step=None,**kw):
            original_log(metrics,step=step,**kw)
            value={'step':int(step or 0),'total_steps':cfg.num_train_steps,'mode':args.mode}
            for k,v in metrics.items():
                try:value[k]=float(v)
                except (TypeError,ValueError):pass
            tmp=output/'progress.tmp';tmp.write_text(json.dumps(value));tmp.replace(output/'progress.json')
        train.wandb.log=log
    train.init_wandb=init_wandb
    original_loader_init=data_loader.TorchDataLoader.__init__
    def bounded(self,*a,**kw):
        original_loader_init(self,*a,**kw)
        if self._data_loader.num_workers:self._data_loader.timeout=180
    data_loader.TorchDataLoader.__init__=bounded
    (output/'training_config.txt').write_text(repr(cfg))
    print('CONFIG',cfg,flush=True)
    train.main(cfg)
    if args.mode=='union':
        final=cfg.checkpoint_dir/'999';assert final.is_dir()
        for item in ['params','assets']:shutil.copytree(final/item,output/item,dirs_exist_ok=True)
    shutil.copy2(cfg.checkpoint_dir/'eval_history.jsonl',output/'eval_history.jsonl')
    for name in ['union-manifest.json','train_union.py','union_data.py','eval_indices_dagger.json','eval_indices_language.json']:
        shutil.copy2(ROOT/name,output/name)
    (output/'completed.json').write_text(json.dumps({'mode':args.mode,'updates':cfg.num_train_steps,'success':True}))
    print('EXPORT_COMPLETE',output,flush=True)

if __name__=='__main__':
    main()
