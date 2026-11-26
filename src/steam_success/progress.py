"""Timestamped status messages, so a long run shows what it is doing."""
import time
from datetime import datetime

_start = time.time()


def status(message: str):
    """Print a message with the clock time and the time since the program started."""
    elapsed = time.time() - _start
    print(f"[{datetime.now():%H:%M:%S} | +{elapsed / 60:4.1f} min] {message}", flush=True)


def banner(text: str):
    line = "=" * 72
    print(f"\n{line}\n{text}\n{line}", flush=True)
