import os
import shutil
import tempfile
from pathlib import Path


def create_video_last_frame(
    video_dir: str,
    video_path: str,
    framerate: int = 30,
) -> None:
    # get the last image in the video dir for each frame
    Path(video_path).parent.mkdir(exist_ok=True, parents=True)
    frame_dirs = list(Path(video_dir).iterdir())
    image_paths = [list(sorted(fd.iterdir()))[-1] for fd in sorted(frame_dirs)]

    # create a temporary directory and copy the files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for i, image_path in enumerate(image_paths):
            # Copy each image to the temp directory, renaming it in the process
            shutil.copy(image_path, temp_path / f"{i:06d}.png")

        args: list[str] = [
            f"ffmpeg -framerate {framerate}",
            f'-pattern_type glob -i "{temp_path / "*.png"}"',
            f"-c:v libx264 -pix_fmt yuv420p {video_path}",
            "-y",
        ]
        os.system(" ".join(args))


def create_video(video_dir: str, video_path: str, framerate: int = 30) -> None:
    Path(video_path).parent.mkdir(exist_ok=True, parents=True)
    args: list[str] = [
        f"ffmpeg -framerate {framerate}",
        f'-pattern_type glob -i "{Path(video_dir) / "*.png"}"',
        f"-c:v libx264 -pix_fmt yuv420p {video_path}",
        "-y",
    ]
    os.system(" ".join(args))
