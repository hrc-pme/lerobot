#!/usr/bin/env python

import argparse
import yaml
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, create_repo
import requests.exceptions

def push_meta_data_to_hub(repo_id: str, meta_data_dir: Path, revision: Optional[str] = None):
    api = HfApi()
    try:
        # 檢查 repository 是否存在
        api.repo_info(repo_id, repo_type="dataset")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 如果 repository 不存在，則自動建立一個新的
            print(f"Repository {repo_id} not found. Creating a new repository...")
            create_repo(repo_id, repo_type="dataset")
        else:
            # 其他 HTTP 錯誤，直接拋出
            raise e

    # 上傳 meta_data
    api.upload_folder(
        folder_path=meta_data_dir,
        path_in_repo="meta_data",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
    )

def push_dataset_card_to_hub(repo_id: str, revision: Optional[str] = None, tags: Optional[list] = None, text: Optional[str] = None):
    from lerobot.common.datasets.utils import create_lerobot_dataset_card
    card = create_lerobot_dataset_card(tags=tags, text=text)
    card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=revision)

def push_videos_to_hub(repo_id: str, videos_dir: Path, revision: Optional[str] = None):
    api = HfApi()
    try:
        # 檢查 repository 是否存在
        api.repo_info(repo_id, repo_type="dataset")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 如果 repository 不存在，則自動建立一個新的
            print(f"Repository {repo_id} not found. Creating a new repository...")
            create_repo(repo_id, repo_type="dataset")
        else:
            # 其他 HTTP 錯誤，直接拋出
            raise e

    # 上傳 videos
    api.upload_folder(
        folder_path=videos_dir,
        path_in_repo="videos",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns="*.mp4",
    )

def push_to_hub(
    repo_id: str,
    local_dir: str,  # 這裡改為 str，並在函數內部轉換為 Path
    push_to_hub: bool = True,
    video: bool = True,
):
    # 將 local_dir 轉換為 Path 物件
    local_dir = Path(local_dir)
    meta_data_dir = local_dir / "meta_data"
    videos_dir = local_dir / "videos"

    if push_to_hub:
        push_meta_data_to_hub(repo_id, meta_data_dir)
        push_dataset_card_to_hub(repo_id)
        if video:
            push_videos_to_hub(repo_id, videos_dir)

def get_args():
    with open("/root/lerobot/lerobot/configs/scripts/push2hub.yaml", "r") as file:
        config = yaml.safe_load(file)
    config = {k.replace("-", "_"): v for k, v in config.items()}
    parser = argparse.ArgumentParser(description="Push LeRobot dataset to Hugging Face Hub.")
    parser.set_defaults(**config)
    return parser.parse_args()

def main():
    args = get_args()
    push_to_hub(**vars(args))

if __name__ == "__main__":
    main()