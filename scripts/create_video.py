import os
import shutil
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from lib.utils.logger import create_logger

log = create_logger("create_video")


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def create_video(cfg: DictConfig) -> None:
    log.info("==> loading settings ...")
    framerate = cfg.get("framerate", 30)
    video_dir = cfg.get("video_dir", None)
    assert video_dir is not None
    video_path = cfg.get("video_path", f"{Path(video_dir).stem}.mp4")

    # get the last image in the video dir for each frame
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


if __name__ == "__main__":
    create_video()
