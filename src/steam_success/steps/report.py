"""Final step — collect the results and figures into one self-contained page: output/report.html.

Inputs : the CSV files in data/derived and the figures in output/figures
Output : output/report.html (open it in any browser; the figures are embedded, so it can be shared as one file)
"""
import base64
import html
from datetime import datetime

import pandas as pd

from steam_success import config
from steam_success.progress import status

FIGURES = [   # (file stem, caption), in reading order
    ("02_target_distributions", "The three success measures across the analysed games."),
    ("02_trends_by_year", "Games released and the success measures by release year."),
    ("02_targets_by_genre", "Success measures by genre (dashed line: all games)."),
    ("02_targets_by_price", "Success measures by regular price."),
    ("03_feature_reach_gap", "Share of games with ≥ 100k owners, with each feature minus without it."),
    ("04_reach_confusion_matrix", "Model A, reach: where each actual tier is predicted."),
    ("04_feature_importance", "Model A: share of XGBoost's total gain, top 10 features per measure."),
    ("05_model_a_vs_b", "Model A vs Model B on the same games and the same 15 folds."),
    ("06_shap_reception_engagement", "SHAP: what moves reception and engagement predictions."),
    ("06_shap_reach", "SHAP: what pushes a game toward 500k+ owners."),
    ("06_shap_model_b", "Model B: share of importance from the early-review features."),
    ("06_high_reach_low_engagement", "High-reach, low-engagement games vs other high-reach games."),
    ("07_baseline_tuning", "Baseline tuning."),
    ("07_search_progress", "XGBoost random search: score of each trial."),
    ("07_search_landscape", "XGBoost random search: score against each setting."),
    ("07_learning_curves_model_a", "Learning curves, Model A."),
    ("07_learning_curves_model_b", "Learning curves, Model A vs Model B on the review sample."),
]

CSS = """
:root { --ink:#16202c; --muted:#5a6472; --line:#dde1e6; --bg:#fbfbf9; --card:#ffffff; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; } h2 { font-size: 20px; margin: 40px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }
h3 { font-size: 16px; margin: 24px 0 8px; } .muted { color: var(--muted); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin:20px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card .num { font-size:26px; font-weight:700; color:var(--accent); } .card .lbl { font-size:13px; color:var(--muted); }
table { border-collapse: collapse; font-size:13.5px; margin: 6px 0 4px; max-width:100%; display:block; overflow-x:auto; width:max-content; }
td { background:var(--card); }
th, td { border-bottom:1px solid var(--line); padding:6px 10px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; } thead th { background:#f1f3f5; }
figure { margin: 18px 0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; } figcaption { font-size:13px; color:var(--muted); margin-top:6px; }
.missing { color:#a14a00; font-size:13px; }
"""


def _csv(name, **kwargs):
    path = config.DERIVED_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _table(df, floatfmt="{:.3f}"):
    if df is None:
        return '<p class="missing">Not available: the step that makes this table has not run.</p>'
    fmt = {c: (lambda v, f=floatfmt: "" if pd.isna(v) else f.format(v))
           for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    return df.to_html(index=False, border=0, formatters=fmt, escape=True, na_rep="")


def _figure(stem, caption):
    path = config.FIG_DIR / f"{stem}.png"
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _cards(summary, comparison, n_games):
    cards = []
    if n_games:
        cards.append((f"{n_games:,}", "games analysed (2015–2023, 20+ reviews)"))
    if summary is not None:
        s = summary.set_index("Measure")
        cards += [(f"{s.loc['Reception', 'XGBoost']:.3f}", "reception R², XGBoost (baseline "
                   f"{s.loc['Reception', 'Baseline']:.3f})"),
                  (f"{s.loc['Reach', 'XGBoost']:.3f}", f"reach macro F1, XGBoost (always 'Low' "
                   f"{s.loc['Reach', 'Reference']:.3f})"),
                  (f"{s.loc['Engagement', 'XGBoost']:.3f}", "engagement R², XGBoost (baseline "
                   f"{s.loc['Engagement', 'Baseline']:.3f})")]
    if comparison is not None:
        r = comparison.set_index("Measure").iloc[0]
        cards.append((r["Gap (B − A)"].split(" ±")[0], f"reach gain from early reviews; B better in {r['B better in']}"))
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="num">{html.escape(n)}</div><div class="lbl">{html.escape(t)}</div></div>'
        for n, t in cards) + "</div>"


def build_report() -> str:
    funnel = _csv("population_funnel.csv")
    summary = _csv("model_a_summary.csv")
    robust = _csv("model_a_robustness.csv")
    comparison = _csv("model_b_comparison.csv")
    shap = _csv("shap_importance.csv")
    hrle = _csv("high_reach_low_engagement_summary.csv")
    holdout = _csv("holdout_comparison.csv")
    growth = _csv("learning_curve_growth.csv")
    corr = _csv("target_correlations.csv")

    n_games = None
    if funnel is not None:
        n_games = int(str(funnel.iloc[-1, 1]).replace(",", ""))
    if funnel is not None:
        funnel = funnel.rename(columns={funnel.columns[0]: "Step"})
    if robust is not None:
        robust = robust.rename(columns={robust.columns[0]: "Features used"})
    if corr is not None:
        corr = corr.rename(columns={corr.columns[0]: ""})
    if hrle is not None:
        hrle = hrle.rename(columns={hrle.columns[0]: ""})

    top_features = ""
    if shap is not None:
        for target, g in shap.groupby("target", sort=False):
            t = g.head(8)[["label", "mean_abs_shap", "direction"]].rename(
                columns={"label": "Feature", "mean_abs_shap": "Mean |SHAP|", "direction": "Direction (rank corr.)"})
            top_features += f"<h3>{html.escape(target)}</h3>" + _table(t)

    figures = "".join(_figure(stem, cap) for stem, cap in FIGURES)
    holdout_cols = None if holdout is None else holdout[[c for c in holdout.columns if c != "Score"]]

    body = f"""
<h1>What Makes a Steam Game Successful?</h1>
<p class="muted">Results of the analysis pipeline · generated {datetime.now():%Y-%m-%d %H:%M} ·
Yasin Erfani Pour, Evan Dos Santos · ECE 9603/9063</p>
<p>Three success measures (reception = share of positive reviews, reach = owner tier, engagement = median
playtime) are predicted from 42 launch features (<b>Model A</b>). <b>Model B</b> adds 8 features from each
game's first 30 days of reviews, compared with Model A on the same games and folds.</p>
{_cards(summary, comparison, n_games)}

<h2>1. Data</h2>
<h3>Analysis population (Table I)</h3>{_table(funnel)}
<h3>How the three measures relate (Spearman correlation)</h3>{_table(corr, "{:.2f}")}

<h2>2. Model A: launch features only</h2>
<h3>Out-of-fold scores, 5-fold cross-validation (Table III)</h3>{_table(summary)}
<h3>Robustness: without features that can change after launch</h3>{_table(robust)}

<h2>3. Model B: adding the first 30 days of reviews</h2>
<h3>Model A vs Model B, mean ± SD over 15 paired folds (Table IV)</h3>{_table(comparison)}

<h2>4. What drives the predictions (SHAP)</h2>{top_features or _table(None)}
<h3>High reach, low engagement (≥ 100k owners, bottom quarter of playtime)</h3>{_table(hrle, "{:,.2f}")}

<h2>5. Tuning and learning curves</h2>
<h3>Default vs tuned models on the 20% hold-out set (Table VI)</h3>{_table(holdout_cols)}
<h3>What doubling the training data adds</h3>{_table(growth)}

<h2>6. Figures</h2>{figures or '<p class="missing">No figures found.</p>'}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steam Game Success: Results</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"""


def main():
    status("Building the results report ...")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "report.html"
    path.write_text(build_report(), encoding="utf-8")
    status(f"Report saved: {path}")
    return path


if __name__ == "__main__":
    main()
