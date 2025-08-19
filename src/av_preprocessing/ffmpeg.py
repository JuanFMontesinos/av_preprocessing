import os
import re
import subprocess
from typing import List, Literal, Tuple


def get_duration_fps(filename: str, display: Literal["s", "ms", "min", "h"]) -> Tuple[float, float]:
    """
    Wraps ffprobe to get file duration and fps
    :param filename: str, Path to file to be evaluate
    :param display: ['ms','s','min','h'] Time format miliseconds, sec, minutes, hours.
    :return: tuple(time, fps) in the mentioned format
    """

    def ffprobe2ms(time: str) -> List[int]:
        cs = int(time[-2::])
        s = int(os.path.splitext(time[-5::])[0])
        idx = time.find(":")
        h = int(time[0 : idx - 1])
        m = int(time[idx + 1 : idx + 3])
        return [h, m, s, cs]

    # Get length of video with filename
    time = None
    fps = None
    result = subprocess.Popen(["ffprobe", str(filename)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    info_lines = [x for x in output if "Duration:" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration:" in x]
    fps_line = [x for x in info_lines if "Stream" in x]
    if duration_line:
        duration_str = duration_line[0].split(",")[0]
        pattern = "\d{2}:\d{2}:\d{2}.\d{2}"
        dt = re.findall(pattern, duration_str)[0]
        time = ffprobe2ms(dt)
    if fps_line:
        pattern = "(\d{2})(.\d{2})* fps"
        fps_elem = re.findall(pattern, fps_line[0])[0]
        fps = float(fps_elem[0] + fps_elem[1])
    if display == "s":
        time = time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0
    elif display == "ms":
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) * 1000
    elif display == "min":
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 60
    elif display == "h":
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 3600
    return (time, fps)


def ffmpeg_call(video_path: str, dst_path: str, input_options: list, output_options: list, ext: None) -> None:
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output
    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isdir(os.path.dirname(dst_path))
    if ext is not None:
        dst_path = os.path.splitext(dst_path)[0] + ext
    result = subprocess.Popen(
        ["ffmpeg", *input_options, "-i", video_path, *output_options, dst_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout = result.stdout.read().decode("utf-8")
    stderr = result.stderr
    if stdout != "":
        print(stdout)
    if stderr is not None:
        print(stderr.read().decode("utf-8"))
