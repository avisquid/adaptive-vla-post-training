"""pi0.5 on the RSM cooking data, built only on public OpenPI APIs (no OpenPI patches).

Contents:
- CookingInputs / CookingOutputs: map the dataset's two cameras and 10-d absolute
  end-effector state/action ([xyz, rot6d, gripper]) to pi0.5's inputs, and cut the
  32-d model output back to 10.
- CookingData: the OpenPI data config for the language-annotated export.
- training_dataset: keeps the 223 training episodes and stops action chunks at
  annotated task boundaries.
- rise_real_preprocess: the image augmentation of the cooking recipe.
- SplitLRAdamW: AdamW with a lower learning rate for the SigLIP image encoder.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import augmax
import einops
import jax
import numpy as np
import optax
import pandas as pd
import torch.utils.data
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer

# The exact data sunshine's pi05-kitchen-sft-v5 runs used: byte-identical episodes of
# jmarangola/rsm_cooking_v5 @ ea7caa6 plus seven-task labels, the SFT split and
# segment boundaries (annotation.language_remaining).
REPO_ID = "xinyunsunshine/rsm_cooking_v5_language"
REVISION = "b92545b777ee572d65382ee080b07b7c2ebf2f1c"
TRAIN_FRAMES = 163_850
# Normalization statistics copied from that run's checkpoints (identical for h10 and h50).
ASSET_ID = "local/rsm_cooking_v5_language"
ACTION_DIM = 10


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CookingInputs(_transforms.DataTransformFn):
    """Global camera -> base_0_rgb, wrist -> left_wrist_0_rgb; the right wrist slot is blank and masked."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class CookingOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class CookingData(_config.DataConfigFactory):
    """Absolute actions (no delta transform); the prompt is the frame's canonical task instruction."""

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "observation.images.global",
            "observation/wrist_image": "observation.images.wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(
            inputs=[CookingInputs(model_type=model_config.model_type)], outputs=[CookingOutputs()])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


class SegmentClamp(torch.utils.data.Dataset):
    """Repeat the last action of the current annotated segment instead of crossing into the next task."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        remaining = int(item["annotation.language_remaining"].item())  # Counts the current frame.
        if remaining < 1:
            raise ValueError(f"frame {index} has annotation.language_remaining={remaining}")
        if remaining < len(item["action"]):
            action = item["action"].clone()
            action[remaining:] = action[remaining - 1]
            item["action"] = action
        return item


def split_indices(root: Path, split: str) -> list[int]:
    """Global frame indices of the episodes meta/language_split.json assigns to `split` (train | valid)."""
    assignment = json.loads((root / "meta/language_split.json").read_text())["episodes"]
    episodes = {int(e) for e, v in assignment.items() if v["split"] == split}
    frames = pd.concat([pd.read_parquet(p, columns=["index", "episode_index"])
                        for p in sorted((root / "data").glob("*/*.parquet"))]).sort_values("index")
    # Subset indexes the full LeRobotDataset by position, so positions must equal the index column.
    if not np.array_equal(frames["index"].to_numpy(), np.arange(len(frames))):
        raise ValueError(f"{root}: the frame index column is not 0..N-1")
    return frames.loc[frames["episode_index"].isin(episodes), "index"].tolist()


def dataset_root(repo_id: str) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return Path(LeRobotDatasetMetadata(repo_id).root)


def training_dataset(create_torch_dataset):
    """Wrap OpenPI's create_torch_dataset: training episodes only, chunks clamped to their segment."""

    def create(data_config, action_horizon, model_config):
        dataset = create_torch_dataset(data_config, action_horizon, model_config)
        indices = split_indices(dataset_root(data_config.repo_id), "train")
        if len(indices) != TRAIN_FRAMES:
            raise ValueError(f"expected {TRAIN_FRAMES} training frames, found {len(indices)}")
        return torch.utils.data.Subset(SegmentClamp(dataset), indices)

    return create


AUG_CROP_FRAC = 0.01


def rise_real_preprocess(preprocess_observation):
    """Replace OpenPI's training augmentation with the cooking recipe's "rise_real" mode.

    OpenPI's default crops 95 % and rotates +-5 deg on the global camera and jitters colour
    with augmax's default hue of +-0.1 on every camera. rise_real crops 1 % (then resizes
    back) and jitters brightness/contrast/saturation with hue 0 on every camera, without
    rotation: hue tells the tomato, pan and bowl apart. Inference (train=False) is unchanged.
    """

    def preprocess(rng, observation, *, train=False, **kwargs):
        observation = preprocess_observation(rng, observation, train=False, **kwargs)
        if not train:
            return observation
        images = {}
        for key, image in observation.images.items():
            image = image / 2.0 + 0.5  # [-1, 1] -> [0, 1] for augmax.
            height, width = image.shape[1:3]
            chain = augmax.Chain(
                augmax.RandomCrop(int(round(width * (1 - AUG_CROP_FRAC))), int(round(height * (1 - AUG_CROP_FRAC)))),
                augmax.Resize(width, height),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.0),
            )
            # Same key for every camera, as in OpenPI's own augmentation.
            image = jax.vmap(chain)(jax.random.split(rng, image.shape[0]), image)
            images[key] = image * 2.0 - 1.0
        return observation.replace(images=images)

    return preprocess


def _path_string(path) -> str:
    parts = []
    for key in path:
        for attribute in ("key", "name", "idx"):
            if hasattr(key, attribute):
                parts.append(str(getattr(key, attribute)))
                break
    return "/".join(parts)


@dataclasses.dataclass(frozen=True)
class SplitLRAdamW(_optimizer.AdamW):
    """AdamW whose parameters matching `encoder_regex` (the SigLIP tower, PaliGemma/img) train at
    lr * encoder_lr_scale. Gradient clipping stays one global clip over all parameters."""

    encoder_lr_scale: float = 0.2
    encoder_regex: str = ".*img.*"

    def labels(self, params):
        pattern = re.compile(self.encoder_regex)
        return jax.tree_util.tree_map_with_path(
            lambda path, _: "encoder" if pattern.fullmatch(_path_string(path)) else "rest", params)

    def create(self, lr, weight_decay_mask=None):
        def adamw(scale):
            return optax.adamw(lambda step: lr(step) * scale, b1=self.b1, b2=self.b2, eps=self.eps,
                               weight_decay=self.weight_decay, mask=weight_decay_mask)

        return optax.chain(
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.multi_transform({"encoder": adamw(self.encoder_lr_scale), "rest": adamw(1.0)}, self.labels),
        )
