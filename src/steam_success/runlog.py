"""A log file that mirrors everything the program prints.

Every normal run saves a copy of its console output (status updates, tables, warnings and
error messages) to its own file in the `logs/` folder, for example
`logs/run_2026-11-27_14-05-31.log`. The console looks exactly as before: the log is an
extra copy, not a replacement.

How it works, in plain terms: Python sends everything that `print()` shows to a "stream"
called `sys.stdout` (and error messages to `sys.stderr`). While a run is logged, we swap
each stream for a small stand-in (`_Tee`) that forwards every message to the real console
AND to the log file. No other code needs to change, because the whole project already uses
`print()` (directly, or through `progress.status()`).

Typical use (see pipeline.main):

    with RunLog():
        run_the_analysis()
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

from steam_success import config

# The log file of the run that is in progress (None when nothing is being logged).
_current_path: Path | None = None


def log_path() -> Path | None:
    """Path of the log file for the current run, or None if this run is not being logged."""
    return _current_path


class _Tee:
    """Stands in for sys.stdout / sys.stderr and sends every message to two places."""

    def __init__(self, console, log_file):
        self._console = console   # the real screen (or whatever the stream pointed to before)
        self._log = log_file      # the open log file

    def write(self, text):
        """Show the text on the console and append it to the log file."""
        written = self._console.write(text)   # console first, so the screen never depends on the log
        try:
            self._log.write(text)
            self._log.flush()   # flush now, so the file is complete even if the program crashes later
        except (OSError, ValueError):
            pass   # a full disk or a closed file must never stop the analysis
        return written

    def flush(self):
        """Flush both destinations (print(..., flush=True) calls this)."""
        self._console.flush()
        try:
            self._log.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        # Anything else a library asks of a stream (isatty, encoding, fileno ...) is answered by
        # the real console, so code that checks "is this a terminal?" keeps working.
        return getattr(self._console, name)


class RunLog:
    """Context manager that logs everything printed inside its `with` block.

    The log file is created when the block starts and closed when it ends, even if the run
    stops because of an error.
    """

    def __init__(self, description: str = ""):
        self.description = description   # shown in the log header, e.g. the command that was run
        self.path: Path | None = None
        self._file = None
        self._saved_streams = None

    # -- start ---------------------------------------------------------------
    def __enter__(self):
        global _current_path
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.path = self._new_log_path()
            # buffering=1 = write line by line; utf-8 because the output contains symbols such as ≥ and ×
            self._file = open(self.path, "w", encoding="utf-8", errors="replace", buffering=1)
        except OSError as e:
            # A read-only folder, for example. The analysis matters more than its log.
            print(f"Warning: could not create a log file in {config.LOG_DIR} ({e}); "
                  "continuing without one.", file=sys.stderr)
            self.path = self._file = None
            return self

        # Swap in the two-way streams. print() looks up sys.stdout every time it runs,
        # so from here on everything is copied to the file.
        self._saved_streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._file)
        sys.stderr = _Tee(sys.stderr, self._file)
        _current_path = self.path

        self._file.write(f"Run started : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                         f"Command     : {self.description}\n"
                         f"Python      : {sys.version.split()[0]}\n"
                         f"{'-' * 72}\n")
        return self

    # -- end -----------------------------------------------------------------
    def __exit__(self, exc_type, exc, tb):
        global _current_path
        if self._file is None:   # logging never started, nothing to undo
            return False

        # An error that nobody handled: Python itself prints it to the console after we return,
        # so write it to the log only (otherwise it would appear twice on screen).
        # SystemExit (a normal "stop the program") is not an error and is skipped here.
        if exc_type is not None and not issubclass(exc_type, SystemExit):
            self._file.write("".join(traceback.format_exception(exc_type, exc, tb)))

        outcome = "finished normally" if exc_type is None else f"stopped ({exc_type.__name__})"
        self._file.write(f"{'-' * 72}\nRun ended   : {datetime.now():%Y-%m-%d %H:%M:%S}, {outcome}\n")

        # Put the original console streams back, then close the file.
        sys.stdout, sys.stderr = self._saved_streams
        self._file.close()
        self._file = None
        _current_path = None
        return False   # False = do not swallow the exception; let it carry on as before

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _new_log_path() -> Path:
        """A file name made from the current time; a number is added if that name is taken."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = config.LOG_DIR / f"run_{stamp}.log"
        extra = 2
        while path.exists():   # two runs started within the same second
            path = config.LOG_DIR / f"run_{stamp}_{extra}.log"
            extra += 1
        return path
