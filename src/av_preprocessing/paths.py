from pathlib import Path
from typing import List, Optional, Iterable, Union

AV_PREPROCESSING_LIB_PATH = Path(__file__).resolve().parent
ASSETS_PATH = AV_PREPROCESSING_LIB_PATH.parents[1] / "assets"
LANDMARK_LIB_PATH = AV_PREPROCESSING_LIB_PATH.parents[1] / "3DDFA_V2"
AV_HUBERT_LIB_PATH = AV_PREPROCESSING_LIB_PATH.parents[1] / "av_hubert"

_default_video_extensions: List[str] = [".mp4", ".mpg"]
_default_audio_extensions: List[str] = [".wav", ".mp3"]


def list_of_video_files(
    directory: Union[str, Path],
    video_ext: Optional[Iterable[str]] = None,
) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Normalize extensions to lowercase with leading dot
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (video_ext or _default_video_extensions)]

    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def list_of_audio_files(
    directory: Union[str, Path],
    audio_ext: Optional[Iterable[str]] = None,
) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Normalize extensions to lowercase with leading dot
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (audio_ext or _default_audio_extensions)]

    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts and "__MACOSX" not in str(p)]
