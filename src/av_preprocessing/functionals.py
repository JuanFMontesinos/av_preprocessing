#av_preprocessing>functionals.py
import logging
from . import ffmpeg


def resample_audio(src_path: str, dst_path: str, target_sr: int, verbose=False):
    # read all the files in folders and subfolders and remove the wav files
    iopts = ["-y"]
    if not verbose:
        iopts = iopts + ["-hide_banner", "-loglevel", "error"]

    oopts = ["-ar", str(target_sr), "-ac", "1"]
    logging.info(f"Calling: ffmpeg {' '.join(iopts)} -i {src_path} {' '.join(oopts)} {dst_path}")
    ffmpeg.ffmpeg_call(src_path, dst_path, iopts, oopts, None)
