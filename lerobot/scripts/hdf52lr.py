#!/usr/bin/env python

import argparse
import yaml
from pathlib import Path
from typing import Any, Optional
import warnings
import shutil
import torch
from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.datasets.compute_stats import compute_stats

def get_from_raw_to_lerobot_format_fn(raw_format: str):
    if raw_format == "pusht_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.pusht_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "umi_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.umi_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "aloha_hdf5":
        from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
    elif "openx_rlds" in raw_format:
        from lerobot.common.datasets.push_dataset_to_hub.openx_rlds_format import from_raw_to_lerobot_format
    elif raw_format == "dora_parquet":
        from lerobot.common.datasets.push_dataset_to_hub.dora_parquet_format import from_raw_to_lerobot_format
    elif raw_format == "xarm_pkl":
        from lerobot.common.datasets.push_dataset_to_hub.xarm_pkl_format import from_raw_to_lerobot_format
    elif raw_format == "cam_png":
        from lerobot.common.datasets.push_dataset_to_hub.cam_png_format import from_raw_to_lerobot_format
    else:
        raise ValueError(f"Unsupported raw format: {raw_format}")

    return from_raw_to_lerobot_format

def save_meta_data(info: dict[str, Any], stats: dict, episode_data_index: dict[str, list], meta_data_dir: Path):
    meta_data_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_data_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    save_file(flatten_dict(stats), meta_data_dir / "stats.safetensors")
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    save_file(episode_data_index, meta_data_dir / "episode_data_index.safetensors")

def convert_dataset(
    raw_dir: Path,
    raw_format: str,
    local_dir: Optional[Path] = None,
    fps: Optional[int] = None,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    episodes: Optional[list[int]] = None,
    force_override: bool = False,
    cache_dir: Path = Path("/tmp"),
    encoding: Optional[dict] = None,
):
    if not raw_dir.exists():
        raise NotADirectoryError(f"Raw directory does not exist: {raw_dir}")

    if local_dir:
        local_dir = Path(local_dir)
        if local_dir.exists():
            if force_override:
                shutil.rmtree(local_dir)
            else:
                raise ValueError(f"Local directory already exists: {local_dir}. Use --force-override.")

        meta_data_dir = local_dir / "meta_data"
        videos_dir = local_dir / "videos"
    else:
        meta_data_dir = cache_dir / "meta_data"
        videos_dir = cache_dir / "videos"

    from_raw_to_lerobot_format = get_from_raw_to_lerobot_format_fn(raw_format)
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir,
        videos_dir=videos_dir,
        fps=fps,
        video=video,
        episodes=episodes,
        encoding=encoding,
    )

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="local",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if local_dir:
        hf_dataset.save_to_disk(str(local_dir / "train"))
        save_meta_data(info, stats, episode_data_index, meta_data_dir)

    return lerobot_dataset

def get_args():
    config_path = "/root/lerobot/lerobot/configs/scripts/push2hub.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = {k.replace("-", "_"): v for k, v in config.items()}
    parser = argparse.ArgumentParser(description="Convert raw dataset to LeRobot format.")
    parser.set_defaults(**config)
    return parser.parse_args()

def main():
    args = get_args()
    convert_dataset(**vars(args))

if __name__ == "__main__":
    main()