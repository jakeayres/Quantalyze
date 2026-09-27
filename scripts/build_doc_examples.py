"""
Run the documentation examples and save what they print and draw.

Every `docs/examples/**/<name>.py` is an example (files starting with `_` are shared
helpers, e.g. the data used on a page). Each example is run twice, once with a light
and once with a dark Matplotlib style, and its results are written next to it:

- `<name>.txt`: everything the example prints
- `<name>.png` and `<name>-dark.png`: the figure it leaves open, if any

The docs pages include the example code, the .txt and the figures, so re-run this
after changing an example or the code it uses:

    uv run python scripts/build_doc_examples.py            # all examples
    uv run python scripts/build_doc_examples.py smoothing  # paths containing "smoothing"
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# Keep __pycache__ folders out of docs/.
sys.dont_write_bytecode = True

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"

# Colours that read well on both the light and the dark docs background.
PALETTE = ["#3f7fd9", "#e0803a", "#2fa58c", "#c2528b", "#8a6fd1"]
THEMES = {
    "light": {"foreground": "#1f2328", "grid": "#d8dee4"},
    "dark": {"foreground": "#e6edf3", "grid": "#3d444d"},
}


def style(theme):
    colors = THEMES[theme]
    return {
        "figure.figsize": (6.4, 3.6),
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
        "font.size": 10,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": colors["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "text.color": colors["foreground"],
        "axes.labelcolor": colors["foreground"],
        "axes.edgecolor": colors["foreground"],
        "xtick.color": colors["foreground"],
        "ytick.color": colors["foreground"],
    }


def example_files(pattern=""):
    return sorted(
        path for path in EXAMPLES.rglob("*.py")
        if not path.name.startswith("_") and pattern in path.as_posix()
    )


def run_example(path):
    """Run one example and return (printed output, open figures)."""
    path = Path(path)
    # Each docs folder has its own `_data.py`; forget the one imported by the last example.
    for name, module in list(sys.modules.items()):
        if str(EXAMPLES) in str(getattr(module, "__file__", "") or ""):
            del sys.modules[name]
    plt.close("all")
    stdout = io.StringIO()
    sys.path.insert(0, str(path.parent))
    try:
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.path.remove(str(path.parent))
    return stdout.getvalue(), [plt.figure(number) for number in plt.get_fignums()]


def build(path):
    for theme in THEMES:
        with matplotlib.rc_context(style(theme)):
            output, figures = run_example(path)
            if len(figures) > 1:
                raise RuntimeError(f"{path.name} leaves {len(figures)} figures open; draw one figure per example.")
            if theme == "light" and output:
                path.with_suffix(".txt").write_text(output, encoding="utf-8")
            suffix = "" if theme == "light" else "-dark"
            for figure in figures:
                figure.savefig(path.with_name(f"{path.stem}{suffix}.png"))
            plt.close("all")


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    for path in example_files(pattern):
        print(path.relative_to(EXAMPLES))
        build(path)
