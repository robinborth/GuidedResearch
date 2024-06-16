import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path


def create_logger(name: str = __name__) -> logging.Logger:
    """
    Create and configure a custom logger with the given name.

    Parameters:
        name (str): The name of the logger. It helps identify the logger when used in
            different parts of the application.

    Returns:
        logging.Logger: A configured logger object that can be used to log messages.

    Usage:
        Use this function to create custom loggers with different names and settings
        throughout your application. Each logger can be accessed using its unique name.

    Example:
        >>> my_logger = create_logger("my_logger")
        >>> my_logger.debug("This is a debug message")
        >>> my_logger.info("This is an info message")
        >>> my_logger.warning("This is a warning message")
        >>> my_logger.error("This is an error message")
        >>> my_logger.critical("This is a critical message")
    """
    # Create a logger with the given name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create a log message formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Return the configured logger
    return logger


def create_video(video_dir: str, video_path: str, framerate: int = 30) -> None:
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
