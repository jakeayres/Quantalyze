"""Every example in the docs runs, and the output shown in the docs is up to date."""
import importlib.util
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_doc_examples", ROOT / "scripts" / "build_doc_examples.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

STALE = "{} is out of date; run `uv run python scripts/build_doc_examples.py` and commit the result."


@pytest.mark.parametrize(
    "path", builder.example_files(), ids=lambda path: path.relative_to(builder.EXAMPLES).as_posix()
)
def test_example(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some examples deliberately show a failing fit
        output, figures = builder.run_example(path)

    if output:
        text = path.with_suffix(".txt")
        assert text.exists() and text.read_text(encoding="utf-8") == output, STALE.format(text.name)
    for suffix in ["", "-dark"] if figures else []:
        image = path.with_name(f"{path.stem}{suffix}.png")
        assert image.exists(), STALE.format(image.name)
