import os
import subprocess


def ffmpeg_call(
    video_path: str, dst_path: str, input_options: list, output_options: list, ext: None
) -> None:
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
