import json
import os

from dropout_profiler import cli
from dropout_profiler.evaluate import run_evaluation


def test_demo_command(capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "informative" in out
    assert "diagnosis" in out


def test_no_command_returns_2():
    assert cli.main([]) == 2


def test_evaluate_command_writes_files(tmp_path, monkeypatch):
    small = run_evaluation(seeds=(301,), n_train=10, n_test=8)
    monkeypatch.setattr(cli, "run_evaluation", lambda: small)
    out_dir = str(tmp_path / "out")
    assert cli.main(["evaluate", "--out", out_dir]) == 0
    for name in ("evaluation.json", "evaluation.csv",
                 "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    with open(os.path.join(out_dir, "evaluation.json")) as fh:
        payload = json.load(fh)
    assert "headline" in payload
    assert payload["n_runs"] == 1
    with open(os.path.join(out_dir, "evaluation.csv")) as fh:
        rows = fh.read().strip().splitlines()
    # header + 3 regimes x 4 imputers x 4 predictors + 4 complete rows
    assert len(rows) == 1 + 48 + 4
