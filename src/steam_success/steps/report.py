"""Final step: collect the results and figures into one self-contained page, output/report.html.

Inputs : the CSV files in data/derived and the figures in output/figures
Output : output/report.html (open it in any browser; the figures are embedded in the file,
         so it can be shared on its own)

How the page is laid out
    * a header with the title, the authors and five headline numbers
    * a sticky menu that jumps to each of the five sections
    * every section holds its own tables and figures, side by side where that fits on the
      screen, instead of keeping all the figures in one long list at the end
    * click a figure to enlarge it (Esc or a click closes it)

The page is plain HTML and CSS written as text below. Nothing is downloaded when it is opened.
If a step has not run yet, its tables show "Not available" and its figures are left out.
"""
import base64
import html
import re
import struct
from datetime import datetime

import pandas as pd

from steam_success import config
from steam_success.progress import status

AUTHORS = "Evan Dos Santos, Yasin Erfani Pour"
COURSE = "ECE 9603/9063"

# Every figure the report can show: file name (without .png) -> caption.
FIGURES = {
    "02_target_distributions": "The three success measures across the analysed games.",
    "02_trends_by_year": "Games released and the success measures by release year.",
    "02_targets_by_genre": "Success measures by genre (dashed line: all games).",
    "02_targets_by_price": "Success measures by regular price.",
    "03_feature_reach_gap": "Share of games with ≥ 100k owners, with each feature minus without it.",
    "04_reach_confusion_matrix": "Model A, reach: where each actual tier is predicted.",
    "04_feature_importance": "Model A: share of XGBoost's total gain, top 10 features per measure.",
    "05_model_a_vs_b": "Model A vs Model B on the same games and the same 15 folds.",
    "06_shap_reception_engagement": "SHAP: what moves reception and engagement predictions.",
    "06_shap_reach": "SHAP: what pushes a game toward 500k+ owners.",
    "06_shap_model_b": "Model B: share of importance from the early-review features.",
    "06_high_reach_low_engagement": "High-reach, low-engagement games vs other high-reach games.",
    "07_baseline_tuning": "Baseline tuning: score for each penalty value tried (mean ± SD over 3 folds; "
                          "ring = selected value; dashed line = default).",
    "07_search_progress": "XGBoost random search: score of each trial.",
    "07_search_landscape": "XGBoost random search: score against each setting.",
    "07_learning_curves_model_a": "Learning curves for Model A: test score of both models (top) and XGBoost "
                                  "training vs test score (bottom).",
    "07_learning_curves_model_b": "Learning curves, Model A vs Model B on the review sample.",
}

# ---------------------------------------------------------------------------
# Style sheet and the small script that enlarges figures
# ---------------------------------------------------------------------------
CSS = """
:root { --ink:#16202c; --muted:#5a6472; --line:#e2e6eb; --bg:#f5f6f8; --card:#ffffff;
        --accent:#2a78d6; --accent-soft:rgba(42,120,214,.16); --pos:#1f6fcf; --neg:#d9531e; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
.wrap { max-width:1400px; margin:0 auto; padding:0 28px; }

/* header */
.hero { background:var(--card); border-bottom:1px solid var(--line); padding:34px 0 30px; }
.hero h1 { margin:0 0 6px; font-size:32px; letter-spacing:-.01em; }
.byline { margin:0 0 14px; color:var(--muted); font-size:14px; }
.byline b { color:var(--ink); font-weight:600; }
.lede { margin:0 0 22px; max-width:100ch; font-size:16px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:14px; }
.kpi { background:var(--bg); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.kpi .num { font-size:30px; font-weight:700; line-height:1.1; color:var(--accent); }
.kpi .title { margin-top:6px; font-weight:600; font-size:13.5px; }
.kpi .sub { color:var(--muted); font-size:12.5px; }

/* menu */
nav.toc { position:sticky; top:0; z-index:10; background:rgba(255,255,255,.94); backdrop-filter:blur(6px); border-bottom:1px solid var(--line); }
nav.toc .wrap { display:flex; gap:6px; overflow-x:auto; padding-top:8px; padding-bottom:8px; }
nav.toc a { white-space:nowrap; text-decoration:none; color:var(--muted); font-size:14px; font-weight:500; padding:6px 12px; border-radius:8px; }
nav.toc a:hover { background:var(--bg); color:var(--ink); }

/* sections */
main { padding-bottom:56px; }
section { padding-top:36px; scroll-margin-top:52px; }
h2 { display:flex; align-items:center; gap:12px; margin:0 0 6px; font-size:24px; letter-spacing:-.01em; }
h2 .n { display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:50%; background:var(--accent); color:#fff; font-size:15px; font-weight:700; }
.intro { margin:0 0 20px; max-width:100ch; color:var(--muted); }
.body { display:grid; grid-template-columns:minmax(0, 1fr); gap:18px; }   /* minmax(0,..) lets wide tables scroll inside their card instead of stretching the page */
h3.sub { margin:12px 0 -4px; font-size:17px; }

/* rows of cards and figures */
.cols { display:grid; gap:18px; align-items:stretch; }
.cols-2 { grid-template-columns:repeat(2, minmax(0, 1fr)); }
.cols-3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
.cols-split { grid-template-columns:minmax(0, 1.2fr) minmax(0, 1fr); align-items:start; }
.stack { display:grid; grid-template-columns:minmax(0, 1fr); gap:18px; }
@media (max-width:1000px) { .cols-2, .cols-3, .cols-split { grid-template-columns:minmax(0, 1fr); } }

/* cards and tables */
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.card h3 { margin:0 0 4px; font-size:15px; }
.note { margin:0 0 10px; color:var(--muted); font-size:12.5px; line-height:1.45; }
.tablewrap { overflow-x:auto; }
table.t { width:100%; border-collapse:collapse; font-size:13.5px; font-variant-numeric:tabular-nums; }
.t th { padding:6px 10px; text-align:left; vertical-align:bottom; font-size:11.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); border-bottom:2px solid var(--line); }
.t td { padding:7px 10px; border-bottom:1px solid var(--line); }
.t tbody tr:last-child td { border-bottom:0; }
.t tbody tr:hover { background:#f8fafc; }
.t .r { text-align:right; }
.t tr.grp td { border-top:2px solid var(--line); }
.pos { color:var(--pos); font-weight:600; }
.neg { color:var(--neg); font-weight:600; }
.diag { color:var(--muted); }
.barbox { position:relative; min-width:96px; padding:1px 8px; text-align:right; }
.barfill { position:absolute; left:0; top:0; bottom:0; background:var(--accent-soft); border-radius:4px; }
.barbox b { position:relative; font-weight:500; }
.missing { margin:0; color:#a14a00; font-size:13px; }

/* figures */
figure.fig { margin:0; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 10px 8px; }
figure.fig img { display:block; width:100%; height:auto; cursor:zoom-in; }
figure.fig figcaption { padding:6px 6px 2px; color:var(--muted); font-size:12.5px; }
#lb { position:fixed; inset:0; z-index:50; display:flex; align-items:center; justify-content:center; padding:24px; background:rgba(15,20,28,.84); cursor:zoom-out; }
#lb[hidden] { display:none; }
#lb img { max-width:100%; max-height:100%; background:#fcfcfb; border-radius:8px; }

footer.wrap { padding-top:8px; padding-bottom:40px; color:var(--muted); font-size:12.5px; }

@media print {
  nav.toc, #lb { display:none; }
  body { background:#fff; }
  .card, figure.fig, section > h2 { break-inside:avoid; }
}
"""

# Enlarges a figure when it is clicked and closes it on any click or on Esc.
SCRIPT = """
const lb = document.getElementById('lb');
document.addEventListener('click', e => {
  const img = e.target.closest('figure.fig img');
  if (img) { lb.firstElementChild.src = img.src; lb.firstElementChild.alt = img.alt; lb.hidden = false; }
  else { lb.hidden = true; }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.hidden = true; });
"""

# A cell like "0.437 ± 0.018", "+0.060", "125,855", "98.6%" or "$14.95" counts as a number
# (it is right-aligned); anything else, such as "R²" or "15 of 15 folds", is text.
NUMBER_LIKE = re.compile(r"^[+\-−–$]?\d[\d,]*(\.\d+)?(%|\s*±.*)?$")


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------
def _csv(name, **kwargs):
    """Read a CSV from data/derived, or return None if the step that makes it has not run."""
    path = config.DERIVED_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _png_size(raw: bytes) -> tuple[int, int]:
    """Width and height of a PNG, read from its header (bytes 16 to 24)."""
    return struct.unpack(">II", raw[16:24])


# ---------------------------------------------------------------------------
# Small HTML building blocks
# ---------------------------------------------------------------------------
def _missing() -> str:
    """The message shown in place of a table whose data file does not exist."""
    return '<p class="missing">Not available: the step that makes this table has not run.</p>'


def _card(title: str, body: str, note: str = "") -> str:
    """A white box with a heading, an optional one-line explanation and the content."""
    note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
    return f'<div class="card"><h3>{html.escape(title)}</h3>{note_html}{body}</div>'


def _cols(kind: str, *items: str) -> str:
    """Place items side by side. kind: cols-2, cols-3 or cols-split (a wider left column)."""
    return f'<div class="cols {kind}">' + "".join(items) + "</div>"


def _stack(*items: str) -> str:
    """Place items one under the other inside one column."""
    return '<div class="stack">' + "".join(items) + "</div>"


def _section(anchor: str, number: int, title: str, intro: str, *blocks: str) -> str:
    """One numbered section of the page: heading, a short introduction, then its blocks."""
    return (f'<section id="{anchor}"><h2><span class="n">{number}</span>{html.escape(title)}</h2>'
            f'<p class="intro">{html.escape(intro)}</p><div class="body">' + "".join(blocks) + "</div></section>")


def _figure(stem: str) -> str:
    """A figure embedded in the page, or nothing if its image file is not there."""
    path = config.FIG_DIR / f"{stem}.png"
    if not path.exists():
        return ""
    raw = path.read_bytes()
    width, height = _png_size(raw)   # stated in the tag so the page does not jump around while loading
    caption = html.escape(FIGURES[stem])
    data = base64.b64encode(raw).decode()   # the image itself, written as text inside the HTML file
    return (f'<figure class="fig"><img alt="{caption}" width="{width}" height="{height}" title="Click to enlarge" '
            f'src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>')


def _format_value(value, fmt: str) -> str:
    """One table cell as text: blank for missing, thousands separators for whole numbers, fmt for decimals."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return fmt.format(value)
    return str(value)


def _sign_class(text: str) -> str:
    """'pos' for a value written with a leading +, 'neg' for a leading minus, otherwise nothing."""
    if text.startswith("+"):
        return "pos"
    if text.startswith(("-", "−")):
        return "neg"
    return ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _table(df, formats=None, default="{:.3f}", signed=(), heat=False, group_cols=0) -> str:
    """A DataFrame as an HTML table.

    formats    {column: format string} for decimals; `default` is used for the other decimal columns
    signed     columns whose values are shown with + or - and coloured by that sign
    heat       shade the number cells by size (used for the correlation table)
    group_cols the first N columns hold repeated labels: show each label only where it changes
    """
    if df is None:
        return _missing()
    formats = formats or {}
    columns = list(df.columns)
    raw_rows = df.values.tolist()

    # 1. Turn every cell into text.
    cells = []
    for row in raw_rows:
        out = []
        for col, value in zip(columns, row):
            if col in formats:        # an explicit format for this column wins
                fmt = formats[col]
            elif col in signed:       # differences are shown with their sign: +0.005 / -0.017
                fmt = "{:+.3f}"
            else:
                fmt = default
            out.append(_format_value(value, fmt))
        cells.append(out)

    # 2. Decide each column's alignment: right-aligned when every filled cell looks like a number.
    #    The first column holds the row labels and stays left-aligned.
    right = [i > 0 and all(NUMBER_LIKE.match(r[i]) for r in cells if r[i] != "") for i in range(len(columns))]

    # 3. Build the HTML.
    head = "".join(f'<th class="{"r" if right[i] else ""}">{html.escape(str(c))}</th>' for i, c in enumerate(columns))
    body = []
    previous = None
    for row_index, (row, raw) in enumerate(zip(cells, raw_rows)):
        shown = list(row)
        starts_group = False
        if group_cols and previous is not None:
            # blank a label that repeats the one above (and everything to its left is also unchanged)
            for i in range(group_cols):
                if all(raw[j] == previous[j] for j in range(i + 1)):
                    shown[i] = ""
            starts_group = raw[0] != previous[0]
        previous = raw

        tds = []
        for i, text in enumerate(shown):
            classes = ["r"] if right[i] else []
            style = ""
            if columns[i] in signed:
                classes.append(_sign_class(text))
            if heat and i > 0 and isinstance(raw[i], float):
                if columns[i] == raw[0]:   # the diagonal: a measure compared with itself
                    classes.append("diag")
                else:
                    style = f' style="background:rgba(42,120,214,{min(abs(raw[i]), 1) * 0.6:.2f})"'
            tds.append(f'<td class="{" ".join(c for c in classes if c)}"{style}>{html.escape(text)}</td>')
        body.append(f'<tr class="{"grp" if starts_group else ""}">' + "".join(tds) + "</tr>")
    return (f'<div class="tablewrap"><table class="t"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _shap_table(group: pd.DataFrame, with_direction: bool) -> str:
    """Top 8 features of one SHAP group: a bar for the importance, and optionally a direction."""
    top = group.head(8)
    peak = top["mean_abs_shap"].max() or 1   # the longest bar is the largest value in this table
    head = '<th>Feature</th><th class="r">Mean |SHAP|</th>' + ('<th class="r">Direction</th>' if with_direction else "")
    rows = []
    for _, r in top.iterrows():
        bar = (f'<td><div class="barbox"><span class="barfill" style="width:{r["mean_abs_shap"] / peak:.0%}"></span>'
               f'<b>{r["mean_abs_shap"]:.3f}</b></div></td>')
        direction = ""
        if with_direction:
            d = r["direction"]
            direction = ("<td></td>" if pd.isna(d) else
                         f'<td class="r {_sign_class(f"{d:+.2f}")}">{d:+.2f}</td>')
        rows.append(f"<tr><td>{html.escape(str(r['label']))}</td>{bar}{direction}</tr>")
    return (f'<div class="tablewrap"><table class="t"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _profile_value(label: str, value) -> str:
    """Format one cell of the high-reach, low-engagement table according to what its row measures."""
    if pd.isna(value):
        return ""
    name = label.lower()
    if "free" in name:        # the two free-to-play rows are shares of games
        return f"{value:.0%}"
    if name == "games":       # a count
        return f"{value:,.0f}"
    if "year" in name:        # a year: no thousands separator
        return f"{value:.0f}"
    if "price" in name:
        return f"${value:,.2f}"
    if "playtime" in name:
        return f"{value:.1f}"
    return f"{value:g}"       # e.g. languages: 6 or 7.5


def _profile_table(df) -> str:
    """The high-reach, low-engagement comparison, with each row formatted for what it measures."""
    if df is None:
        return _missing()
    labels = df.iloc[:, 0].astype(str)
    shown = pd.DataFrame({
        "Group profile": labels,
        "Low-engagement group": [_profile_value(l, v) for l, v in zip(labels, df.iloc[:, 1])],
        "Other high-reach games": [_profile_value(l, v) for l, v in zip(labels, df.iloc[:, 2])],
    })
    return _table(shown)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _kpis(summary, comparison, n_games) -> str:
    """The row of headline numbers at the top of the page."""
    cards = []   # (big number, title, small print)
    if n_games:
        cards.append((f"{n_games:,}", "Games analysed",
                      f"{config.YEAR_MIN}–{config.YEAR_MAX}, {config.MIN_REVIEWS}+ reviews"))
    if summary is not None:
        s = summary.set_index("Measure")
        cards += [
            (f"{s.loc['Reception', 'XGBoost']:.3f}", "Reception (R²)",
             f"XGBoost · baseline {s.loc['Reception', 'Baseline']:.3f}"),
            (f"{s.loc['Reach', 'XGBoost']:.3f}", "Reach (macro F1)",
             f"XGBoost · always 'Low' {s.loc['Reach', 'Reference']:.3f}"),
            (f"{s.loc['Engagement', 'XGBoost']:.3f}", "Engagement (R²)",
             f"XGBoost · baseline {s.loc['Engagement', 'Baseline']:.3f}"),
        ]
    if comparison is not None:
        r = comparison.set_index("Measure").iloc[0]   # the first row is reach
        cards.append((r["Gap (B − A)"].split(" ±")[0], "Reach gain from early reviews",
                      f"Model B vs A · better in {r['B better in']}"))
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="num">{html.escape(n)}</div><div class="title">{html.escape(t)}</div>'
        f'<div class="sub">{html.escape(s)}</div></div>' for n, t, s in cards) + "</div>"


def _lede() -> str:
    """The introduction under the title (the tier names come from config, so they cannot drift)."""
    labels = config.REACH_LABELS
    tiers = ", ".join(labels[:-1]) + " or " + labels[-1]   # "Low (<20k), Moderate (...), High (...) or Very high (...)"
    return (f"Three success measures are predicted from 42 launch features (<b>Model A</b>): "
            f"<b>reception</b>, the share of positive reviews; <b>reach</b>, the owner tier: {html.escape(tiers)}; "
            f"and <b>engagement</b>, the median playtime. <b>Model B</b> adds {len(config.REVIEW_FEATURES)} features "
            f"from each game's first {config.WINDOW_DAYS} days of reviews and is compared with Model A on the "
            f"same games and folds.")


# ---------------------------------------------------------------------------
# The whole page
# ---------------------------------------------------------------------------
def build_report() -> str:
    """Assemble the complete HTML text of the report."""
    funnel = _csv("population_funnel.csv")
    summary = _csv("model_a_summary.csv")
    robust = _csv("model_a_robustness.csv")
    comparison = _csv("model_b_comparison.csv")
    shap = _csv("shap_importance.csv")
    hrle = _csv("high_reach_low_engagement_summary.csv")
    holdout = _csv("holdout_comparison.csv")
    growth = _csv("learning_curve_growth.csv")
    corr = _csv("target_correlations.csv")

    # The number of analysed games is the last row of the population table (its "Titles" column).
    n_games = int(str(funnel.iloc[-1, 1]).replace(",", "")) if funnel is not None else None
    # The CSVs saved with an unnamed first column; give those columns a heading.
    if funnel is not None:
        funnel = funnel.rename(columns={funnel.columns[0]: "Step"})
    if robust is not None:
        robust = robust.rename(columns={robust.columns[0]: "Features used"})
    if corr is not None:
        corr = corr.rename(columns={corr.columns[0]: ""})
    if holdout is not None:
        holdout = holdout[[c for c in holdout.columns if c != "Score"]]   # the metric is named in the section text

    # ---- 1. Data
    data = _section(
        "data", 1, "Data",
        "The games that were analysed, how they were selected, and how the three success measures are distributed.",
        _cols("cols-2",
              _card("Analysis population (Table I)", _table(funnel),
                    "Each row applies one more filter to the full game catalogue."),
              _card("How the three measures relate", _table(corr, default="{:.2f}", heat=True),
                    "Spearman rank correlation: 1 means games rank the same on both measures, "
                    "0 means no relationship.")),
        _figure("02_target_distributions"), _figure("02_trends_by_year"),
        _figure("02_targets_by_genre"), _figure("02_targets_by_price"))

    # ---- 2. Model A
    model_a = _section(
        "model-a", 2, "Model A: launch features only",
        "Each game is predicted by a model that never saw it (5-fold cross-validation). "
        "XGBoost is compared with a simple baseline and with a reference that uses no features.",
        _cols("cols-split",
              _stack(
                  _card("Out-of-fold scores (Table III)", _table(summary),
                        "R² = share of the variation explained (0 = no better than guessing the average). "
                        "Macro F1 = F1 averaged over the four reach tiers. Reference = guess the average "
                        "(always 'Low' for reach). Baseline = ridge or logistic regression."),
                  _card("Robustness: without features that can change after launch", _table(robust),
                        "XGBoost refitted without the language counts and the number of achievements.")),
              _figure("04_reach_confusion_matrix")),
        _figure("04_feature_importance"))

    # ---- 3. Model B
    model_b = _section(
        "model-b", 3, "Model B: adding the first 30 days of reviews",
        "Model A and Model B are trained and tested on exactly the same games and the same 15 folds "
        "(5-fold cross-validation repeated 3 times).",
        _card("Model A vs Model B, mean ± SD over 15 paired folds (Table IV)", _table(comparison),
              "Gap = Model B score minus Model A score on the same folds."),
        _figure("05_model_a_vs_b"))

    # ---- 4. SHAP
    shap_tables = {"a": [], "b": []}
    if shap is not None:
        for target, g in shap.groupby("target", sort=False):
            is_b = target.startswith("Model B")
            notes = {"Reception": "Effect on % positive, in percentage points.",
                     "Engagement": "Effect on log playtime (log hours).",
                     "Reach (Very high tier)": "Importance across all four tiers (log-odds); direction is for "
                                               "the Very high tier.",
                     "Model B Reach": "Importance across all four tiers (log-odds). Orange bars in the figure "
                                      "below are early-review features.",
                     "Model B Engagement": "Effect on log playtime (log hours)."}
            shap_tables["b" if is_b else "a"].append(
                _card(target, _shap_table(g, with_direction=not is_b), notes.get(target, "")))
    if not shap_tables["a"] and not shap_tables["b"]:
        shap_tables["a"] = [_card("Top features", _missing())]

    group_guide = _card(
        "How to read the two charts below",
        "<p class='note' style='margin:0 0 8px'><b>Left:</b> the share of games with ≥ 100k owners among games that "
        "have a feature, minus the share among games without it. Blue bars: more likely to be high-reach; "
        "red bars: less likely. Only yes/no features present in 1–99% of games are shown.</p>"
        "<p class='note' style='margin:0'><b>Right:</b> the share of high-reach, low-engagement games with each "
        "feature, minus the share among other high-reach games, in percentage points. Orange bars: more common "
        "in the low-engagement group; blue bars: less common.</p>")

    drivers = _section(
        "drivers", 4, "What drives the predictions (SHAP)",
        "SHAP measures how much each feature moves a prediction. The bar is the average size of that push "
        "(mean |SHAP|). Direction is the rank correlation between a feature's value and its push: + means higher "
        "values raise the prediction, − means they lower it.",
        '<h3 class="sub">Model A: launch features</h3>',
        _cols("cols-3", *shap_tables["a"]),
        _figure("06_shap_reception_engagement"), _figure("06_shap_reach"),
        '<h3 class="sub">Model B: with early-review features</h3>' if shap_tables["b"] else "",
        _cols("cols-2", *shap_tables["b"]) if shap_tables["b"] else "",
        _figure("06_shap_model_b"),
        '<h3 class="sub">Comparing groups of games</h3>',
        _cols("cols-2",
              _card("High reach, low engagement (≥ 100k owners, bottom quarter of playtime)", _profile_table(hrle),
                    "Medians for each group, except the number of games and the two free-to-play rows, "
                    "which are shares of the group."),
              group_guide),
        _cols("cols-2", _figure("03_feature_reach_gap"), _figure("06_high_reach_low_engagement")))

    # ---- 5. Tuning
    tuning = _section(
        "tuning", 5, "Tuning and learning curves",
        "Default and tuned settings are compared once on a 20% hold-out set that the tuning never saw, with a "
        "bootstrap interval for the difference. Learning curves show whether more training data would help.",
        _card("Default vs tuned models on the hold-out set (Table VI)",
              _table(holdout, signed=("XGBoost: tuned − default", "95% low", "95% high")),
              "Scores are R² for reception and engagement and macro F1 for reach. A positive tuned − default "
              "value means tuning helped; the 95% interval comes from resampling the hold-out games."),
        _card("What doubling the training data adds",
              _table(growth, signed=("Gain from doubling",), group_cols=2),
              "Test score with 50% and with 100% of the training data. Training − test gap = how much better "
              "the model scores on games it learned from than on games it has not seen."),
        _figure("07_baseline_tuning"), _figure("07_search_progress"), _figure("07_search_landscape"),
        _figure("07_learning_curves_model_a"), _figure("07_learning_curves_model_b"))

    nav = "".join(f'<a href="#{a}">{html.escape(t)}</a>' for a, t in
                  [("data", "1 · Data"), ("model-a", "2 · Model A"), ("model-b", "3 · Model B"),
                   ("drivers", "4 · What drives predictions"), ("tuning", "5 · Tuning")])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steam Game Success: Results</title><style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
<h1>What Makes a Steam Game Successful?</h1>
<p class="byline"><b>{html.escape(AUTHORS)}</b> · {COURSE} · results of the analysis pipeline, generated {datetime.now():%Y-%m-%d %H:%M}</p>
<p class="lede">{_lede()}</p>
{_kpis(summary, comparison, n_games)}
</div></header>
<nav class="toc"><div class="wrap">{nav}</div></nav>
<main class="wrap">{data}{model_a}{model_b}{drivers}{tuning}</main>
<footer class="wrap">Figures are embedded in this file; click one to enlarge it.</footer>
<div id="lb" hidden><img alt=""></div>
<script>{SCRIPT}</script>
</body></html>"""


def main():
    """Write output/report.html."""
    status("Building the results report ...")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "report.html"
    path.write_text(build_report(), encoding="utf-8")
    status(f"Report saved: {path}")
    return path


if __name__ == "__main__":
    main()
