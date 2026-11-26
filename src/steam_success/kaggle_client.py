"""Kaggle access shared by the steps that download review files."""
import getpass
import os
import time


class RateLimited(Exception):
    """Kaggle kept answering HTTP 429 (too many requests)."""


def connect():
    """Return an authenticated Kaggle API client.

    The token is read from the KAGGLE_API_TOKEN environment variable, or asked for once.
    It is never written to disk by this code.
    """
    if not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = getpass.getpass("Kaggle API token: ").strip()
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported late: it checks credentials on import

    api = KaggleApi()
    api.authenticate()
    return api


def with_retry(fn, label: str, waits=(0, 30, 60, 120, 240, 300)):
    """Call fn(), retrying with back-off when Kaggle rate-limits the request."""
    for wait in waits:
        if wait:
            print(f"Rate-limited on {label}; retrying in {wait}s")
            time.sleep(wait)
        try:
            return fn()
        except Exception as e:  # the Kaggle client raises plain exceptions
            if "429" not in str(e) and "Too Many Requests" not in str(e):
                raise
    raise RateLimited(f"Rate limit persisted for {label}; try again later.")
