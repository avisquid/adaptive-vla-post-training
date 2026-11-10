"""A pinned, episode-disjoint union with source-specific prompts and chunk bounds."""
import dataclasses
import json
from pathlib import Path
import numpy as np
import torch
from openpi.training import data_loader

ROOT=Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'union-manifest.json').read_text())
ORIGINAL_CREATE=data_loader.create_torch_dataset
REPOS={'dagger':'avisquid/DAgger_1','language':'xinyunsunshine/rsm_cooking_v5_language'}

class SegmentClamp:
    def __init__(self, dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        item=dict(self.dataset[index])
        n=int(item['annotation.language_remaining'].item())
        assert n>=1
        action=item['action'].clone()
        if n<len(action):
            action[n:]=action[n-1].clone()
        item['action']=action
        return item


def source_dataset(name,split,config,horizon,model):
    dc=dataclasses.replace(config,repo_id=REPOS[name],episodes=MANIFEST['splits'][name][split])
    dataset=ORIGINAL_CREATE(dc,horizon,model)
    return SegmentClamp(dataset) if name=='language' else dataset


def union_dataset(config,horizon,model):
    assert config.repo_id=='local/cooking_union'
    return torch.utils.data.ConcatDataset([source_dataset(name,'train',config,horizon,model) for name in ['dagger','language']])


def evaluation_loader(config,split,sharding,n_batches,seed=0):
    name='dagger' if split=='valid' else 'language'
    assert split in ['valid','language_valid']
    dc=config.data.create(config.assets_dirs,config.model)
    raw=source_dataset(name,'valid',dc,config.model.action_horizon,config.model)
    dataset=data_loader.transform_dataset(raw,dc)
    n=min(len(dataset),n_batches*config.batch_size)
    indices=sorted(np.random.default_rng(seed).choice(len(dataset),n,replace=False).tolist())
    path=ROOT/f'eval_indices_{name}.json'
    if path.exists():
        assert json.loads(path.read_text())==indices
    else:
        path.write_text(json.dumps(indices))
    loader=data_loader.TorchDataLoader(torch.utils.data.Subset(dataset,indices),local_batch_size=config.batch_size,
        sharding=sharding,num_batches=n//config.batch_size,num_workers=2,seed=seed)
    return data_loader.DataLoaderImpl(dc,loader)
