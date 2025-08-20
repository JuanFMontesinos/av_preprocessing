"""
File: extract_landmarks_from_a_dataset.py
Author: Juan Montesinos
Created: 19/08/2025

Extracts face landmarks from all videos in a dataset and saves then in a specified directory. Additionally, 
it extracts metadata that containes information about the video processing and the number of frames.

How to run:
    uv run scripts/extract_landmarks_from_a_dataset.py <dataset_dir> [--dst_dir <destination_dir>] [--save_videos]
Example:
    uv run scripts/extract_landmarks_from_a_dataset.py /mnt/DataNMVE/grid_dataset
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fire import Fire

import av_preprocessing as avp


def process_dataset(dataset_dir: str, dst_dir: Optional[str] = None, save_videos: bool = False):
    dataset_path = Path(dataset_dir)  # type: ignore
    if dst_dir is None:
        dst_path = dataset_path
    else:
        dst_path = Path(dst_dir)

    ld_dst_dir = dst_path / "landmarks"
    videos_dst_dir = dst_path / "check_videos"

    video_files = avp.paths.list_of_video_files(dataset_dir)
    n_video_files = len(video_files)

    logging.info(f"Found {n_video_files} video files in {dataset_dir}")
    logging.info(
        f"Landmarks will be saved to {ld_dst_dir}"
        f" and videos will be saved to {videos_dst_dir if save_videos else 'not saved'}"
    )
    for i, video_file in enumerate(video_files):
        logging.info(f"[{i + 1}/{n_video_files}] Processing {video_file.name}")
        failed, metadata = avp.functionals.extract_landmarks(
            video_path=video_file,
            video_dst=videos_dst_dir / video_file.relative_to(dataset_dir) if save_videos else None,
            landmarks_dst=ld_dst_dir / video_file.relative_to(dataset_dir).with_suffix(".npy"),
            metadata_dst=ld_dst_dir / video_file.relative_to(dataset_dir).with_suffix(".yaml"),
        )
        if failed:
            logging.error(f"Failed to process {video_file.name}. Skipping.")
            continue


if __name__ == "__main__":
    # debug mode
    if len(sys.argv) < 2:
        process_dataset("/mnt/DataNMVE/grid_dataset", save_videos=True)
    else:
        Fire(process_dataset)
