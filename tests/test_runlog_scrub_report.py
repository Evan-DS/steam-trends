"""Checks of the log file, the --scrub command and the report page (no data files needed, runs in seconds)."""
import sys

import pandas as pd
import pytest

from steam_success import config, pipeline, runlog, scrub
from steam_success.steps import report


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A small fake project in a temporary folder, with every config path pointed at it.

    Layout: data/source (input files), data/derived and output (generated files, each with a
    .gitkeep placeholder), and a few cache folders. Nothing touches the real project.
    """
    root = tmp_path / "proj"
    for folder in ("data/source", "data/derived/downloads", "output/figures", "src/pkg/__pycache__"):
        (root / folder).mkdir(parents=True)
    for placeholder in ("data/source/.gitkeep", "data/derived/.gitkeep", "output/.gitkeep"):
        (root / placeholder).write_text("")
    (root / "data/source/games.parquet").write_text("INPUT DATA")
    (root / "data/derived/games_clean.parquet").write_text("derived")
    (root / "data/derived/downloads/temp.csv").write_text("temp")
    (root / "output/report.html").write_text("<html></html>")
    (root / "output/figures/fig.png").write_text("png")
    (root / "src/pkg/__pycache__/mod.pyc").write_text("pyc")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache/README.md").write_text("cache")

    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "SOURCE_DIR", root / "data/source")
    monkeypatch.setattr(config, "DERIVED_DIR", root / "data/derived")
    monkeypatch.setattr(config, "OUTPUT_DIR", root / "output")
    monkeypatch.setattr(config, "FIG_DIR", root / "output/figures")
    monkeypatch.setattr(config, "LOG_DIR", root / "logs")
    return root


# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
def test_everything_printed_reaches_screen_and_log(project, capsys):
    with runlog.RunLog("python main.py --test") as log:
        print("a status line")
        print("an error line", file=sys.stderr)
    shown = capsys.readouterr()
    assert "a status line" in shown.out and "an error line" in shown.err   # the screen is unchanged
    text = log.path.read_text(encoding="utf-8")
    assert "a status line" in text and "an error line" in text             # and the log has a copy
    assert "python main.py --test" in text and "finished normally" in text
    assert log.path.parent == project / "logs"


def test_log_records_an_error_and_streams_are_restored(project):
    before = (sys.stdout, sys.stderr)
    with pytest.raises(ValueError):
        with runlog.RunLog("failing run") as log:
            raise ValueError("boom")
    assert (sys.stdout, sys.stderr) == before   # the console is back to normal
    assert runlog.log_path() is None
    text = log.path.read_text(encoding="utf-8")
    assert "Traceback" in text and "boom" in text and "stopped (ValueError)" in text


def test_two_runs_in_the_same_second_get_different_files(project, monkeypatch):
    from datetime import datetime

    class FrozenClock:
        """A stand-in for datetime whose now() always returns the same moment."""
        @staticmethod
        def now():
            return datetime(2026, 11, 27, 12, 0, 0)

    monkeypatch.setattr(runlog, "datetime", FrozenClock)
    with runlog.RunLog("first") as a:
        pass
    with runlog.RunLog("second") as b:
        pass
    assert a.path.name == "run_2026-11-27_12-00-00.log"
    assert b.path.name == "run_2026-11-27_12-00-00_2.log"   # a number is added instead of overwriting the first log


def test_main_logs_the_message_of_a_failed_run(project, capsys):
    with pytest.raises(SystemExit) as stop:
        pipeline.main(["--steps", "99x"])
    assert stop.value.code == 1
    assert "Unknown step" in capsys.readouterr().err        # still shown on screen
    (log_file,) = (project / "logs").glob("run_*.log")
    assert "Unknown step '99x'" in log_file.read_text(encoding="utf-8")   # and saved in the log


def test_list_and_scrub_do_not_create_a_log(project):
    pipeline.main(["--list"])
    pipeline.main(["--scrub"])
    assert not (project / "logs").exists()


# ---------------------------------------------------------------------------
# --scrub
# ---------------------------------------------------------------------------
def test_scrub_removes_generated_files_and_keeps_inputs(project):
    with runlog.RunLog("make a log"):
        print("hello")
    assert (project / "logs").exists()

    assert scrub.scrub() is True

    assert not (project / "data/derived/games_clean.parquet").exists()
    assert not (project / "data/derived/downloads").exists()
    assert not (project / "output/report.html").exists()
    assert not (project / "output/figures").exists()
    assert not (project / "logs").exists()                          # emptied, and removed as it is not in git
    assert not (project / "src/pkg/__pycache__").exists()
    assert not (project / ".pytest_cache").exists()
    # kept: the input data and the placeholders that keep the folders in git
    assert (project / "data/source/games.parquet").read_text() == "INPUT DATA"
    for placeholder in ("data/source/.gitkeep", "data/derived/.gitkeep", "output/.gitkeep"):
        assert (project / placeholder).exists()


def test_scrub_twice_reports_a_clean_project(project, capsys):
    scrub.scrub()
    capsys.readouterr()
    assert scrub.scrub() is True
    assert "already clean" in capsys.readouterr().out


def test_scrub_never_empties_a_folder_outside_the_project(project, tmp_path, monkeypatch, capsys):
    outside = tmp_path / "somewhere_else"
    outside.mkdir()
    (outside / "important.txt").write_text("keep me")
    monkeypatch.setattr(config, "DERIVED_DIR", outside)   # e.g. a mistyped STEAM_DERIVED_DIR
    scrub.scrub()
    assert (outside / "important.txt").exists()
    assert "not inside the project folder" in capsys.readouterr().out


def test_scrub_never_empties_a_folder_that_contains_the_source_data(project, monkeypatch, capsys):
    monkeypatch.setattr(config, "DERIVED_DIR", project / "data")   # would include data/source
    scrub.scrub()
    assert (project / "data/source/games.parquet").exists()
    assert "contains the source data" in capsys.readouterr().out


def test_scrub_cannot_be_combined_with_other_options(project):
    with pytest.raises(SystemExit) as stop:
        pipeline.main(["--scrub", "--steps", "3"])
    assert stop.value.code == 2   # argparse's "bad command line" code
    assert (project / "output/report.html").exists()   # nothing was deleted


# ---------------------------------------------------------------------------
# Report page
# ---------------------------------------------------------------------------
def test_report_builds_without_any_results_and_shows_names_in_order(project):
    page = report.build_report()   # no CSVs and no figures exist in the fake project
    assert "Evan Dos Santos, Yasin Erfani Pour" in page
    assert "Yasin Erfani Pour, Evan Dos Santos" not in page
    assert "Not available" in page   # tables say so instead of crashing


def test_report_includes_a_figure_next_to_its_section(project):
    # a minimal valid PNG header is enough: only its width and height are read
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (640).to_bytes(4, "big") + (320).to_bytes(4, "big")
           + b"\x08\x06\x00\x00\x00")
    (config.FIG_DIR / "05_model_a_vs_b.png").write_bytes(png)
    page = report.build_report()
    assert 'width="640" height="320"' in page
    # the figure sits inside section 3 (Model B), before section 4 starts
    assert page.index('id="model-b"') < page.index("data:image/png") < page.index('id="drivers"')


def test_profile_table_formats_each_row_for_what_it_measures():
    assert report._profile_value("Games", 1335.0) == "1,335"
    assert report._profile_value("Release year", 2018.0) == "2018"          # no thousands separator in a year
    assert report._profile_value("Free to play (price $0)", 0.4157) == "42%"
    assert report._profile_value("Regular price ($)", 1.98) == "$1.98"
    assert report._profile_value("Languages supported", 6.0) == "6"


def test_table_colours_signs_and_blanks_repeated_labels():
    df = pd.DataFrame({"Target": ["A", "A", "B"], "Model": ["x", "y", "x"], "Gain": [0.01, -0.02, 0.0]})
    page = report._table(df, signed=("Gain",), group_cols=1)
    assert 'class="r pos"' in page and 'class="r neg"' in page
    assert page.count(">A<") == 1   # the second "A" is blanked
