from __future__ import annotations

from thermal_model.cli import main


def test_cli_no_args_prints_help(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "thermal-model" in captured.out
