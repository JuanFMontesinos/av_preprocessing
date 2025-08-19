"""
File: resample_audio_dataset.py
Author: Juan Montesinos
Created: 19/08/2025

Resamples all audio files in a directory to a specified sample rate and mono channel, keeping the folder structure intact.
How to use:
    uv run scripts/resample_audio_dataset.py <dataset_dir> [<dst_dir>] [<sample_rate>]

"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fire import Fire

import av_preprocessing as avp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def resample_audio_dataset(
    dataset_dir: str,
    dst_dir: Optional[str] = None,
    sample_rate: int = 16000,
):
    dataset_dir = Path(dataset_dir)

    if dst_dir is None:
        dst_dir = dataset_dir.parent / f"audio_{sample_rate}"
    else:
        dst_dir = Path(dst_dir)

    audio_files = avp.paths.list_of_audio_files(dataset_dir)
    n_audio_files = len(audio_files)
    
    for i, file in enumerate(audio_files):
        # keep relative path to preserve folder structure
        rel_path = file.relative_to(dataset_dir)
        dst_file = dst_dir / rel_path

        # create output directory if needed
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        avp.functionals.resample_audio(file.as_posix(), dst_file.as_posix(), sample_rate, verbose=False, prefix=f"[{i+1}/{n_audio_files}] ")


if __name__ == "__main__":
    # debug mode
    if len(sys.argv) < 2:
        resample_audio_dataset(
            "/mnt/DataNMVE/grid_dataset/audio_25k",
            "/mnt/DataNMVE/grid_dataset/audio_16k",
            16000,
        )
    else:
        Fire(resample_audio_dataset)
