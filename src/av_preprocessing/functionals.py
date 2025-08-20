# av_preprocessing>functionals.py
import logging
import os
import importlib.util
import sys
from collections import deque
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any, Union, List

import torch
import imageio
import numpy as np
import yaml

from . import ffmpeg, paths, av_hubert


def resample_audio(src_path: str, dst_path: str, target_sr: int, verbose=False, prefix: str = ""):
    # read all the files in folders and subfolders and remove the wav files
    iopts = ["-y"]
    if not verbose:
        iopts = iopts + ["-hide_banner", "-loglevel", "error"]

    oopts = ["-ar", str(target_sr), "-ac", "1"]
    logging.info(f"{prefix} Calling: ffmpeg {' '.join(iopts)} -i {src_path} {' '.join(oopts)} {dst_path}")
    ffmpeg.ffmpeg_call(src_path, dst_path, iopts, oopts, None)


def extract_landmarks(
    video_path: Union[str, Path],
    video_dst: Optional[Union[str, Path]] = None,
    landmarks_dst: Optional[Union[str, Path]] = None,
    metadata_dst: Optional[Union[str, Path]] = None,
    config_file=paths.LANDMARK_LIB_PATH / "configs" / "mb1_120x120.yml",
    onnx=True,
    n_pre=1,
    n_next=1,
    start_frame=-1,
    end_frame=-1,
    landmark_type: Literal["2d_sparse", "2d_dense", "3d"] = "2d_sparse",
    assert_fps=None,
    assert_frames=True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run 3DDFA/FaceBoxes landmark tracking over a video with simple temporal smoothing.

    Args:
        video_path: Path to the input video file.
        video_dst: If provided, path where an .mp4 with landmarks overlaid will be written.
        landmarks_dst: If provided, path where landmark arrays will be saved as a .npy stack
            of shape (T, K, 2|3) depending on `landmark_type`. Values are rounded and saved
            as int16 (same as original implementation).
        metadata_dst: If provided, path to save metadata (duration/fps). NOTE: this function
            currently prepares a metadata dict but does not serialize it (kept for parity).
        config_file: YAML config for the landmark library (3DDFA_V2). `checkpoint_fp` and
            `bfm_fp` are injected/overridden based on `LANDMARK_LIB_PATH`.
        onnx: Use ONNX inference (recommended). If False, falls back to the GPU/PyTorch path.
        n_pre: Number of *previous* frames included in the smoothing window.
        n_next: Number of *future* frames included in the smoothing window (look-ahead).
        start_frame: If > 0, skip frames before this index.
        end_frame: If > 0, stop after this index (inclusive).
        landmark_type: One of {'2d_sparse', '2d_dense', '3d'}.
        assert_fps: If provided, assert that the probed FPS equals this value.
        assert_frames: If True, check that duration*fps ≈ true frame count (only when
            processing the full video; i.e., when start_frame<0 and end_frame<0).

    Returns:
        False on success, True on early failure (kept for compatibility with upstream usage).

    Side effects:
        - Optionally writes an annotated .mp4.
        - Optionally writes landmarks .npy (int16).
        - Leaves a `metadata` BaseDict prepared in-memory (not persisted here).

    Notes:
        - Temporal smoothing uses a simple moving average over a window of size n_pre+n_next+1.
        - The first face is detected, then subsequent frames are tracked by cropping around
          previous landmarks. If the tracked ROI becomes too small, face detection is retried.
        - If no face is found in the very first frame considered, the function returns True.
    """
    sys.path.append(str(paths.LANDMARK_LIB_PATH))
    assert landmark_type in ["2d_sparse", "2d_dense", "3d"], (
        f"Landmarks should be either 2d_sparse, 2d_dense or 3d but {landmark_type} found."
    )

    video_path = Path(video_path)
    save_video, save_landmarks, save_metadata = False, False, False
    if video_dst is not None:
        video_dst = Path(video_dst)
        save_video = True
        ext = video_dst.suffix.lower()
        if "mp4" not in ext:
            logging.warning(f"Only mp4 video format supported, converting {ext} to .mp4")
        video_dst = video_dst.with_suffix(".mp4")
        if not video_dst.parent.exists():
            video_dst.parent.mkdir(parents=True, exist_ok=True)
    if landmarks_dst is not None:
        landmarks_dst = Path(landmarks_dst)
        save_landmarks = True
        ext = landmarks_dst.suffix.lower()
        if "npy" not in ext:
            logging.warning(f"Only npy landmarks format supported, converting {ext} to .npy")
        landmarks_dst = landmarks_dst.with_suffix(".npy")
        if not landmarks_dst.parent.exists():
            landmarks_dst.parent.mkdir(parents=True, exist_ok=True)
    if metadata_dst is not None:
        metadata_dst = Path(metadata_dst)
        save_metadata = True
        ext = metadata_dst.suffix.lower()
        if "yaml" not in ext:
            logging.warning(f"Only yaml metadata format supported, converting {ext} to .yaml")
        metadata_dst = metadata_dst.with_suffix(".yaml")
        if not metadata_dst.parent.exists():
            metadata_dst.parent.mkdir(parents=True, exist_ok=True)

    with config_file.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["checkpoint_fp"] = paths.LANDMARK_LIB_PATH.joinpath("weights", "mb1_120x120.pth").as_posix()
    cfg["bfm_fp"] = paths.LANDMARK_LIB_PATH.joinpath("configs", "bfm_noneck_v3.pkl").as_posix()

    from utils.functions import cv_draw_landmark
    from utils.render import render

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if onnx:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        from TDDFA import TDDFA

        tddfa = TDDFA(gpu_mode="gpu", **cfg)
        face_boxes = FaceBoxes()
    duration, fps = ffmpeg.get_duration_fps(video_path, "s")
    expected_frames = int(duration * fps)

    reader = imageio.get_reader(video_path)
    if assert_fps is not None:
        assert fps == assert_fps, f"FPS required to be {assert_fps} but video is {fps} FPS"
    if save_video:
        writer = imageio.get_writer(video_dst, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = n_pre, n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()
    landmarks = []
    # run
    dense_flag = landmark_type in (
        "2d_dense",
        "3d",
    )
    pre_ver = None
    initial_frame = True
    for i, frame in enumerate(reader):  # type: ignore
        if start_frame > 0 and i < start_frame:
            continue

        if end_frame > 0 and i > end_frame:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if initial_frame:
            initial_frame = False
            # detect
            boxes = face_boxes(frame_bgr)  # xmin, ymin, xmax, ymax, score
            if len(boxes) == 0:
                return True, {}
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())
            if save_video:
                for _ in range(n_pre):
                    queue_frame.append(frame_bgr.copy())
                queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy="landmark")

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)

                boxes = [boxes[0]]

                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            queue_ver.append(ver.copy())
            if save_video:
                queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            landmarks.append(ver_ave)

            if save_video:
                if landmark_type == "2d_sparse":
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
                elif landmark_type == "2d_dense":
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
                elif landmark_type == "3d":
                    img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                else:
                    raise ValueError(f"Unknown opt {landmark_type}")

                writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB
                queue_frame.popleft()
            queue_ver.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())

        # the last frame

        ver_ave = np.mean(queue_ver, axis=0)
        landmarks.append(ver_ave)
        if save_video:
            queue_frame.append(frame_bgr.copy())
            if landmark_type == "2d_sparse":
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif landmark_type == "2d_dense":
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif landmark_type == "3d":
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f"Unknown opt {landmark_type}")

            writer.append_data(img_draw[..., ::-1])  # BGR->RGB
            queue_frame.popleft()
        queue_ver.popleft()

    if assert_frames:
        if abs(expected_frames - i) > 5:
            print(f"A duration of {duration} at {fps} fps implies {expected_frames} frames but {i} frames found")
            return True, dict()
    if save_video:
        writer.close()
    if save_landmarks:
        np.save(landmarks_dst, np.stack(landmarks).round().astype(np.int16))  # type: ignore
    if save_metadata:
        metadata = {
            "duration": duration,
            "fps": fps,
            "n_frames": len(landmarks),
            "landmark_type": landmark_type,
            "video_path": str(video_path),
        }
        with metadata_dst.open("w") as f:
            yaml.dump(metadata, f)
    return False, metadata


def crop_rois(
    input_video_path: str, output_path: Union[Path, str], landmarks: List[np.ndarray], mean_face: np.ndarray, fps=25
):
    """
    Crops the video according to the given landmarks
    :param input_video_path: str, Path to input video
    :param output_path: str, Path to output video
    :param landmarks: list, List of  T landmarks of shape 68x2
    """
    STD_SIZE = (256, 256)
    stablePntsIDs = [33, 36, 39, 42, 45]

    rois = av_hubert.crop_patch(
        input_video_path,
        landmarks,
        mean_face,
        stablePntsIDs,
        STD_SIZE,
        window_margin=12,
        start_idx=48,
        stop_idx=68,
        crop_height=96,
        crop_width=96,
    )
    imageio.mimwrite(output_path, [x for x in np.flip(rois, axis=-1)], fps=fps)  # Flip to convert bgr into rgb
    return rois


def extract_visual_feature(video: Union[str, np.ndarray],model,task, max_length=250, **kwargs) -> torch.Tensor:
    # The model has been trained with sequences of 20s (500 frames)
    # Hypothesize using larger sequences may be harmful
    # https://www.juanmontesinos.com/av-inp-dev/data_analysis/visual_features.html
    """
    video: [str,np.array] Path to video or np.array of frames
    """
    spec = importlib.util.spec_from_file_location(
        "avhubert_utils", paths.AV_HUBERT_LIB_PATH / "avhubert" / "utils.py"
    )
    avhubert_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(avhubert_utils)
    model.eval()
    transform = avhubert_utils.Compose(
        [
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
        ]
    )
    if isinstance(video, str):
        frames = avhubert_utils.load_video(video)
        logging.info(f"\tLoad video {video}: shape {frames.shape}")
    else:
        frames = video
    frames = transform(frames)
    logging.info(f"\tCenter crop video to: {frames.shape}")
    temp_feats = frames.shape[0]
    frames: torch.Tensor = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0)
    if torch.cuda.is_available():
        frames = frames.cuda()
    output = torch.empty(temp_feats, 768, dtype=frames.dtype, device=frames.device)
    model = model.encoder.w2v_model
    with torch.no_grad():
        # split the frames into chunks of size max_length
        for i in range(0, temp_feats, max_length):
            end = min(i + max_length, temp_feats)
            feature, _ = model.extract_finetune(
                source={"video": frames[:, :, i:end], "audio": None}, padding_mask=None, output_layer=None
            )
            output[i:end] = feature
        feature = feature.squeeze(dim=0)

    return output
