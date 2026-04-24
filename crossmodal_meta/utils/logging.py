from __future__ import annotations

from pathlib import Path
from typing import Optional


class SimpleLogger:
    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        print(message)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

