# pyrefly: ignore-errors

from pathlib import Path
from typing import List, Tuple, Union
from random import randint

import imageio
import numpy as np
import polars as pl
import torch
import yaml
from scipy.io import wavfile

# from . import paths


class GRIDAudioVisualDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        load_av_hubert_feats: bool,
        load_video_roi: bool,
        min_corruption_time: float,
        max_corruption_time: float,
    ):
        """
        Args:
            dataset_path (str or Path): Path to the dataset directory.
            load_av_hubert_feats (bool): Whether to load AV-Hubert features.
            load_video_roi (bool): Whether to load video ROIs.
            min_corruption_time (float): Minimum corruption time in seconds.
            max_corruption_time (float): Maximum corruption time in seconds.
        """
        self.dataset_path = Path(dataset_path)
        self.landmarks_path = self.dataset_path / "landmarks"
        self.av_hubert_feats_path = self.dataset_path / "av_hubert_feats"
        self.video_roi_path = self.dataset_path / "rois"

        self.load_av_hubert_feats = load_av_hubert_feats
        self.load_video_roi = load_video_roi
        self.min_corruption_time = min_corruption_time
        self.max_corruption_time = max_corruption_time

        self.av_hubert_paths = paths.list_of_npy_files(self.av_hubert_feats_path)
        metadata_list = []

        # This is specific for the GRID dataset
        for p in self.av_hubert_paths:
            metadata_path = self.landmarks_path / p.relative_to(self.av_hubert_feats_path).with_suffix(".yaml")
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            internal_path = p.relative_to(self.av_hubert_feats_path)
            metadata["track_id"] = internal_path.stem
            metadata["speaker_id"] = internal_path.parent.stem
            metadata_list.append(metadata)
        self.metadata_df = pl.DataFrame(metadata_list).with_columns(
            (pl.col("speaker_id") + "@" + pl.col("track_id")).alias("uuid")
        )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int):
        output = {
            "audio_clean": None,
            "audio_masked": None,
            "av_hubert_feats": None,
            "video_roi": None,
            "seed": randint(0, 2**32 - 1),
            "uuid": self.metadata_df[idx]["uuid"],
        }
        rng = np.random.default_rng(output["seed"])
        
        av_feats_path = self.av_hubert_paths[idx]
        audio_path = (
            self.dataset_path / "audio_16kHz" / av_feats_path.relative_to(self.av_hubert_feats_path).with_suffix(".wav")
        )
        roi_path = self.video_roi_path / av_feats_path.relative_to(self.av_hubert_feats_path).with_suffix(".npy")

        # Loading Clean Audio and Masked (corrupted) Audio
        # =========================================================
        sr, audio_clean = wavfile.read(audio_path)
        
        if audio_clean.dtype == np.int16:
            audio_clean = audio_clean.astype(np.float32) / 32768.0
        elif audio_clean.dtype == np.int32:
            audio_clean = audio_clean.astype(np.float32) / 2147483648.0
        elif audio_clean.dtype == np.uint8:
            # uint8 WAV often offset by 128
            audio_clean = (audio_clean.astype(np.float32) - 128.0) / 128.0
        else:
            audio_clean = audio_clean.astype(np.float32, copy=False)

        audio_masked = audio_clean.copy()
        min_elems  = int(self.min_corruption_time * sr)
        max_elems  = min(int(self.max_corruption_time * sr), len(audio_masked))

        corrupt_len = int(rng.integers(min_elems, max_elems + 1))
        start = int(rng.integers(0, max(1, len(audio_masked) - corrupt_len + 1)))
        end = start + corrupt_len
        audio_masked[start:end] = 0.0  # Masking the audio segment
        
        if min_elems > len(audio_masked):
            audio_masked = np.zeros_like(audio_masked)  # The whole track is removed

        output["audio_clean"] = torch.from_numpy(audio_clean)
        output["audio_masked"] = torch.from_numpy(audio_masked)
        # =========================================================
        # Loading AV-Hubert Features
        # =========================================================
        if self.load_av_hubert_feats:
            av_hubert_feats = torch.from_numpy(np.load(av_feats_path).astype("float32"))
            output["av_hubert_feats"] = av_hubert_feats
        # =========================================================
        # Loading Video ROI
        # =========================================================
        if self.load_video_roi:
            video_roi = np.stack(imageio.mimread(roi_path))
            video_roi = torch.from_numpy(video_roi.astype("float32"))
            output["video_roi"] = video_roi
        
        return output
    
if __name__ == "__main__":
    import paths
    # Example usage
    dataset = GRIDAudioVisualDataset(
        dataset_path="/mnt/DataNMVE/grid_test",
        load_av_hubert_feats=True,
        load_video_roi=True,
        min_corruption_time=0.16,
        max_corruption_time=1.6
    )
    
    dataset.__getitem__(0)  # Fetch the first item to test
    print("Dataset initialized and first item fetched successfully.")
        
else:
    from . import paths