"""Chart style and figure saving shared by every step."""
import matplotlib

matplotlib.use("Agg")  # write figures to files; never open windows
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

from steam_success import config  # noqa: E402

# The colour palette used in every figure
COLOR = "#2a78d6"          # main series
COLOR_A = "#86b6ef"        # Model A
COLOR_B = "#104281"        # Model B
COLOR_REVIEW = "#eb6834"   # early-review features / highlighted group
NEG = "#e34948"            # negative values
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"   # text: strongest to faintest
SURFACE = "#fcfcfb"        # the background of every figure

# One shared look: off-white background, light grid, no top/right frame lines, bold left-aligned titles
STYLE = {
    "figure.dpi": 110, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "axes.axisbelow": True,
}
plt.rcParams.update(STYLE)   # apply it to every figure made after this import

# Axis label formatters: 0.8 -> "80%", 2.5 -> "2.5 h", 12 -> "+12 pp" (percentage points), 20000 -> "20k"
percent = mtick.FuncFormatter(lambda v, _: f"{round(v * 100, 2):g}%")
hours = mtick.FuncFormatter(lambda v, _: f"{v:g} h")
signed_pp = mtick.FuncFormatter(lambda v, _: f"{v:+.0f} pp")
thousands = mtick.FuncFormatter(lambda v, _: f"{v / 1000:g}k")


def nice_ticks(axis):
    """About six round-number ticks on an axis (steps of 1, 2, 2.5, 5 or 10)."""
    axis.set_major_locator(mtick.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))


def log_axis(axis, subs=(1, 2, 5)):
    """Log scale with plain-number tick labels (0.01, 0.02, 0.05, ...)."""
    axis.axes.set_xscale("log")
    axis.set_major_locator(mtick.LogLocator(subs=subs))
    axis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mtick.NullFormatter())   # no labels on the small in-between ticks


def save_figure(fig, name: str):
    """Save a figure as output/figures/<name>.png and close it."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")   # tight = no wasted white border
    plt.close(fig)   # free the memory; many figures are made in one run
    print(f"[figure] {path}")
    return path
