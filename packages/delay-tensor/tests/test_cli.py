from pathlib import Path

import pytest

from delay_tensor.cli import main

CONFIG = Path(__file__).parents[1] / "configs" / "tensor_8x64.yaml"


def test_classify_command_writes_a_run(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "classify",
            "--config",
            str(CONFIG),
            "--seeds",
            "7",
            "--samples-per-class",
            "8",
            "--time-samples",
            "16",
            "--out",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "experiment   : classify (1 seeds, tensor_8x64)" in captured.out
    assert "report       :" in captured.out
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    assert (runs[0] / "report.html").exists()
    # The command line is echoed into the report so a run can be reproduced.
    assert "--config" in (runs[0] / "report.html").read_text()


def test_quiet_mode_prints_only_the_directory(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "xor",
            "--config",
            str(CONFIG),
            "--seeds",
            "7",
            "--samples",
            "32",
            "--time-samples",
            "16",
            "--quiet",
            "--out",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip().endswith("-xor")
    assert "report" not in captured.out


def test_bad_config_exits_with_code_two(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("tensor:\n  wires: 1\n  slots: 4\n  slot_dt_ns: 1.0\n  loop_period_ns: 4.0\n")
    code = main(["classify", "--config", str(broken), "--seeds", "7", "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err


def test_missing_config_file_exits_with_code_two(tmp_path: Path, capsys) -> None:
    code = main(["classify", "--config", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_malformed_seeds_exit_with_code_two(tmp_path: Path, capsys) -> None:
    code = main(["classify", "--config", str(CONFIG), "--seeds", "one,two"])
    captured = capsys.readouterr()
    assert code == 2
    assert "comma-separated" in captured.err


def test_empty_seed_list_exits_with_code_two(tmp_path: Path, capsys) -> None:
    code = main(["classify", "--config", str(CONFIG), "--seeds", " , "])
    assert code == 2
    assert "at least one seed" in capsys.readouterr().err


def test_unknown_experiment_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2
