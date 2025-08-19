from pathlib import Path
from typing import List, Optional, Iterable

_default_video_extensions: List[str] = [".mp4", ".mpg"]  # did you mean .mpg (not .mgp)?
_default_audio_extensions: List[str] = [".wav", ".mp3"]


def list_of_video_files(
    directory: str | Path,
    video_ext: Optional[Iterable[str]] = None,
) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Normalize extensions to lowercase with leading dot
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (video_ext or _default_video_extensions)]

    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def list_of_audio_files(
    directory: str | Path,
    audio_ext: Optional[Iterable[str]] = None,
) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # Normalize extensions to lowercase with leading dot
    exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (audio_ext or _default_audio_extensions)]

    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts and "__MACOSX" not in str(p)]

