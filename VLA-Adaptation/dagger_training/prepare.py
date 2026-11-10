"""Download pinned inputs; optionally fetch the completed DAgger-only baseline."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-baseline', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for manifest, destination in [('dagger-1/source.json', 'dagger-1/dataset'),
                                  ('union-comparison/language-source.json', 'union-comparison/language')]:
        ref = json.loads((root / manifest).read_text())
        snapshot_download(ref['repo'], repo_type='dataset', revision=ref['revision'], local_dir=root / destination)
    ref = json.loads((root / 'dagger-1/base-source.json').read_text())
    snapshot_download(ref['repo'], revision=ref['revision'], allow_patterns=[ref['path'] + '**'], local_dir=root / 'dagger-1/base')
    if args.with_baseline:
        HfApi().sync_bucket('hf://buckets/rsm-bucket/rsm-shared/dagger_1_pi05_h50/step_1000', str(root / 'dagger-1/export'))


if __name__ == '__main__':
    main()
