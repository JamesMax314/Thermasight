from __future__ import annotations

from pathlib import Path

import pytest

from thermal_model.cli import main


def test_cli_no_args_prints_help(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "thermal-model" in captured.out


def test_preview_help_works(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as excinfo:
        main(["preview", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--dem" in captured.out
    assert "--what" in captured.out


def test_preview_rejects_invalid_what(
    capsys,  # type: ignore[no-untyped-def]
    synthetic_dem_path: Path,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["preview", "--dem", str(synthetic_dem_path), "--what", "bogus"])
    assert excinfo.value.code == 2


def test_preview_requires_dem(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as excinfo:
        main(["preview"])
    assert excinfo.value.code == 2


def test_preview_save_writes_png_for_convergence(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "convergence.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "convergence",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.parametrize("what", ["slope", "aspect", "curvature", "all"])
def test_preview_save_writes_png_for_each_view(
    synthetic_dem_path: Path, tmp_path: Path, what: str
) -> None:
    out = tmp_path / f"{what}.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            what,
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0
