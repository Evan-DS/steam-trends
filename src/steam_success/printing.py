"""Printing tables to the terminal."""
import pandas as pd

# Print wide tables on one line instead of wrapping them
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def print_table(table, title: str | None = None, formats: dict | str | None = None):
    """Print a DataFrame/Series, optionally with a title and column formats.

    formats: one format string for every cell (e.g. "{:.3f}"), or {column: format or function}.
    """
    if title:
        print(f"\n{title}")
        print("-" * len(title))   # an underline as long as the title
    if isinstance(table, pd.Series):
        table = table.to_frame()   # a Series prints best as a one-column table
    if formats is not None:
        table = table.copy()   # never change the caller's table
        # one format string applies to every column; a dict applies only to the columns it names
        cols = table.columns if isinstance(formats, str) else [c for c in formats if c in table.columns]
        for c in cols:
            fmt = formats if isinstance(formats, str) else formats[c]
            # a ready-made function is used as it is; a format string is wrapped in one (blank for missing values)
            fn = fmt if callable(fmt) else (lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
            table[c] = table[c].map(fn)
    print(table.to_string())
    print()
