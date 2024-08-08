import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from lib.utils.video import create_video

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="optimize")
def main(cfg: DictConfig) -> None:
    log.info("==> loading settings ...")
    framerate = cfg.get("framerate", 30)
    video_dir = cfg.get("video_dir", None)
    assert video_dir is not None
    video_path = cfg.get("video_path", f"{Path(video_dir).stem}.mp4")
    create_video(video_dir=video_dir, video_path=video_path, framerate=framerate)


if __name__ == "__main__":
    main()
