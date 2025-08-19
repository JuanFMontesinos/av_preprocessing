"""
File: resample_audio_dataset.py
Author: Juan Montesinos
Created: 19/08/2025

Resamples all audio files in a directory to a specified sample rate keeping the folder structure intact.
How to use:
    uv run scripts/resample_audio_dataset.py <input_dir> [<output_dir>] [<sample_rate>]

"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fire import Fire

import av_preprocessing as avp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def resample_audio_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 16000,
):
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir.parent / f"audio_{sample_rate}"
    else:
        output_dir = Path(output_dir)

    audio_files = avp.paths.list_of_audio_files(input_dir)

    for file in audio_files:
        # keep relative path to preserve folder structure
        rel_path = file.relative_to(input_dir)
        dst_file = output_dir / rel_path

        # create output directory if needed
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        avp.functionals.resample_audio(file.as_posix(), dst_file.as_posix(), sample_rate, verbose=False)


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
