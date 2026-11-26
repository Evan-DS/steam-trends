"""Printing tables to the terminal."""
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def print_table(table, title: str | None = None, formats: dict | str | None = None):
    """Print a DataFrame/Series, optionally with a title and column formats.

    formats: one format string for every cell (e.g. "{:.3f}"), or {column: format or function}.
    """
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if isinstance(table, pd.Series):
        table = table.to_frame()
    if formats is not None:
        table = table.copy()
        cols = table.columns if isinstance(formats, str) else [c for c in formats if c in table.columns]
        for c in cols:
            fmt = formats if isinstance(formats, str) else formats[c]
            fn = fmt if callable(fmt) else (lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
            table[c] = table[c].map(fn)
    print(table.to_string())
    print()
