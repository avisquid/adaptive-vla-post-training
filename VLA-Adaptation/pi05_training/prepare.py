"""Download the pinned training dataset to $HF_LEROBOT_HOME/xinyunsunshine/rsm_cooking_v5_language (~0.9 GB)."""
from huggingface_hub import snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME

import cooking

if __name__ == "__main__":
    path = snapshot_download(cooking.REPO_ID, repo_type="dataset", revision=cooking.REVISION,
                             local_dir=HF_LEROBOT_HOME / cooking.REPO_ID)
    print(path)
