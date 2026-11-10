"""A process lock shared by the collector and the offline dataset editor."""
import fcntl
from pathlib import Path


class DatasetLock:
    def __init__(self, root):
        root = Path(root).resolve()
        self.path = root.parent / f".{root.name}.dataset.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise RuntimeError("Dataset is in use. Stop collection or wait for the other editor before editing.") from None
        return self

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
