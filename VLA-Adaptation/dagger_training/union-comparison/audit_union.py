import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sources = {'dagger': ROOT.parent / 'dagger-1/dataset', 'language': ROOT / 'language'}
splits = {'dagger': json.loads((ROOT.parent/'dagger-1/split.json').read_text())['episodes'],
          'language': json.loads((ROOT/'language/meta/language_split.json').read_text())['episodes']}
manifest = {'sources': {'dagger': json.loads((ROOT.parent/'dagger-1/source.json').read_text()),
                        'language': json.loads((ROOT/'language-source.json').read_text())},
            'sampling': 'uniform over training frames', 'splits': {}, 'frame_counts': {},
            'episode_hashes': {}, 'boundary_policy': 'repeat last action within annotated segment'}
seen={}
for name, root in sources.items():
    manifest['splits'][name] = {split: sorted(int(k) for k,v in splits[name].items() if v['split']==split) for split in ['train','valid']}
    counts={'train':0,'valid':0};hashes={}
    for p in sorted(root.glob('data/*/*.parquet')):
        ep=int(p.stem.split('_')[-1]);t=pq.read_table(p)
        actions=np.asarray(t['action'].to_pylist(),dtype=np.float32)
        states=np.asarray(t['observation.state'].to_pylist(),dtype=np.float32)
        assert actions.shape==states.shape==(len(t),10)
        assert np.isfinite(actions).all() and np.isfinite(states).all()
        h=hashlib.sha256(actions.tobytes()+states.tobytes()).hexdigest()
        assert h not in seen, ('duplicate episode',seen.get(h),(name,ep))
        seen[h]=(name,ep);hashes[str(ep)]=h
        counts[splits[name][str(ep)]['split']]+=len(t)
        if name=='language':
            tasks=np.asarray(t['task_index'].to_pylist()).reshape(-1)
            remaining=np.asarray(t['annotation.language_remaining'].to_pylist()).reshape(-1)
            expected=np.ones(len(t),dtype=int)
            for i in range(len(t)-2,-1,-1):
                expected[i]=1+expected[i+1] if tasks[i]==tasks[i+1] else 1
            assert np.array_equal(remaining,expected), (ep,'segment boundary mismatch')
    manifest['frame_counts'][name]=counts;manifest['episode_hashes'][name]=hashes
manifest['duplicate_episodes']=0
manifest['total_training_frames']=sum(c['train'] for c in manifest['frame_counts'].values())
(ROOT/'union-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ['frame_counts','total_training_frames','duplicate_episodes']},indent=2))
