from pathlib import Path


class FileStorage:
    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = Path(base_dir)

    def ensure_dir(self, path: str) -> Path:
        directory = self.base_dir / path
        directory.mkdir(parents=True, exist_ok=True)
        return directory

