import logging
import random
import time
from pathlib import Path
from typing import Any
import numpy as np

def get_logger(name: str = "AdaRAG") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

logger = get_logger()

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return round(elapsed, 4)

def ensure_directory(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)

def file_exists(path: Path | str) -> bool:
    return Path(path).exists()

def validate_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    text = text.strip()
    if len(text) == 0:
        raise ValueError("Empty string received.")
    return text

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)