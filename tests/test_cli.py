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


def test_preview_heating_writes_png_with_dem_centre_lat_lon(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    # The synthetic DEM is in EPSG:27700; the CLI should derive lat/lon
    # from its centre via reprojection so the user only has to provide
    # --datetime.
    out = tmp_path / "heating.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "heating",
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_preview_heating_writes_png_with_explicit_lat_lon(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "heating_explicit.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "heating",
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--lat",
            "54.2",
            "--lon",
            "-2.3",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_preview_heating_requires_datetime(synthetic_dem_path: Path) -> None:
    with pytest.raises(SystemExit, match="--datetime"):
        main(
            [
                "preview",
                "--dem",
                str(synthetic_dem_path),
                "--what",
                "heating",
            ]
        )


def test_preview_resolution_speeds_up_a_heating_render(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    # Smoke-test that --resolution drives a coarser render through to
    # the heating pipeline and produces a smaller-but-valid PNG. We
    # don't time-assert here since CI is noisy; the unit test on
    # read_dem already pins the resampling shape semantics.
    out = tmp_path / "heating_2m.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "heating",
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--resolution",
            "2.0",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_preview_resolution_rejects_finer_than_source(
    synthetic_dem_path: Path,
) -> None:
    with pytest.raises(ValueError, match="finer than the source"):
        main(
            [
                "preview",
                "--dem",
                str(synthetic_dem_path),
                "--what",
                "convergence",
                "--resolution",
                "0.5",
            ]
        )


def test_preview_heating_rejects_naive_datetime(synthetic_dem_path: Path) -> None:
    with pytest.raises(SystemExit, match="timezone-naive"):
        main(
            [
                "preview",
                "--dem",
                str(synthetic_dem_path),
                "--what",
                "heating",
                "--datetime",
                "2026-06-21T12:00:00",
            ]
        )


@pytest.mark.parametrize(
    "what",
    ["trigger", "weighted-convergence", "leak", "draft", "cycle-period"],
)
def test_preview_wind_views_write_png(
    synthetic_dem_path: Path, tmp_path: Path, what: str
) -> None:
    """The four wind-requiring preview choices each render a PNG."""
    out = tmp_path / f"{what}.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            what,
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--wind-from",
            "225",
            "--wind-speed",
            "5",
            "--no-resolve-flats",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_run_writes_trigger_leak_and_cycle_outputs(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    """The run subcommand can emit all three rasters when flags are set."""
    trigger_out = tmp_path / "trigger.tif"
    leak_out = tmp_path / "leak.tif"
    cycle_out = tmp_path / "cycle.tif"
    rc = main(
        [
            "run",
            "--dem",
            str(synthetic_dem_path),
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--wind-from",
            "225",
            "--wind-speed",
            "5",
            "--no-resolve-flats",
            "--out",
            str(trigger_out),
            "--leak-out",
            str(leak_out),
            "--cycle-period-out",
            str(cycle_out),
        ]
    )
    assert rc == 0
    assert trigger_out.exists() and trigger_out.stat().st_size > 0
    assert leak_out.exists() and leak_out.stat().st_size > 0
    assert cycle_out.exists() and cycle_out.stat().st_size > 0


def test_run_accepts_draft_aggregation_sigma(
    synthetic_dem_path: Path, tmp_path: Path
) -> None:
    """The new --draft-aggregation-sigma flag is honoured end-to-end."""
    trigger_out = tmp_path / "trigger.tif"
    rc = main(
        [
            "run",
            "--dem",
            str(synthetic_dem_path),
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--wind-from",
            "225",
            "--wind-speed",
            "5",
            "--draft-aggregation-sigma",
            "30",
            "--no-resolve-flats",
            "--out",
            str(trigger_out),
        ]
    )
    assert rc == 0
    assert trigger_out.exists() and trigger_out.stat().st_size > 0


def test_mosaic_cli_writes_output(tmp_path: Path, synthetic_dem_path: Path) -> None:
    # The synthetic_dem fixture is a single tile; mosaic-of-one is a
    # valid (degenerate) call and the smallest end-to-end exercise.
    out = tmp_path / "mosaic.tif"
    rc = main(
        [
            "mosaic",
            "--inputs",
            str(synthetic_dem_path),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_mosaic_cli_refuses_overwrite(tmp_path: Path, synthetic_dem_path: Path) -> None:
    out = tmp_path / "mosaic.tif"
    main(["mosaic", "--inputs", str(synthetic_dem_path), "--output", str(out)])
    with pytest.raises(FileExistsError):
        main(["mosaic", "--inputs", str(synthetic_dem_path), "--output", str(out)])


def test_mosaic_cli_overwrite_flag_works(
    tmp_path: Path, synthetic_dem_path: Path
) -> None:
    out = tmp_path / "mosaic.tif"
    main(["mosaic", "--inputs", str(synthetic_dem_path), "--output", str(out)])
    rc = main(
        [
            "mosaic",
            "--inputs",
            str(synthetic_dem_path),
            "--output",
            str(out),
            "--overwrite",
        ]
    )
    assert rc == 0


def test_preview_heating_with_local_land_cover(
    synthetic_dem_path: Path, synthetic_lcm_path: Path, tmp_path: Path
) -> None:
    """preview --what heating --land-cover PATH renders successfully."""
    out = tmp_path / "heating_with_lcm.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "heating",
            "--land-cover",
            str(synthetic_lcm_path),
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


def test_run_with_local_land_cover_writes_trigger(
    synthetic_dem_path: Path, synthetic_lcm_path: Path, tmp_path: Path
) -> None:
    """run --land-cover PATH passes a per-cell α array through the pipeline."""
    trigger_out = tmp_path / "trigger.tif"
    rc = main(
        [
            "run",
            "--dem",
            str(synthetic_dem_path),
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--wind-from",
            "225",
            "--wind-speed",
            "5",
            "--no-resolve-flats",
            "--land-cover",
            str(synthetic_lcm_path),
            "--out",
            str(trigger_out),
        ]
    )
    assert rc == 0
    assert trigger_out.exists() and trigger_out.stat().st_size > 0


def test_run_rejects_absorptivity_and_land_cover_together(
    synthetic_dem_path: Path, synthetic_lcm_path: Path, tmp_path: Path
) -> None:
    """argparse mutual exclusion enforces one absorptivity source."""
    out = tmp_path / "trigger.tif"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                "--dem",
                str(synthetic_dem_path),
                "--datetime",
                "2026-06-21T12:00:00+01:00",
                "--wind-from",
                "225",
                "--wind-speed",
                "5",
                "--absorptivity",
                "0.7",
                "--land-cover",
                str(synthetic_lcm_path),
                "--out",
                str(out),
            ]
        )
    assert excinfo.value.code == 2


def test_run_rejects_land_cover_and_wms_together(
    synthetic_dem_path: Path, synthetic_lcm_path: Path, tmp_path: Path
) -> None:
    """--land-cover and --land-cover-wms are also mutually exclusive."""
    out = tmp_path / "trigger.tif"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                "--dem",
                str(synthetic_dem_path),
                "--datetime",
                "2026-06-21T12:00:00+01:00",
                "--wind-from",
                "225",
                "--wind-speed",
                "5",
                "--land-cover",
                str(synthetic_lcm_path),
                "--land-cover-wms",
                "--out",
                str(out),
            ]
        )
    assert excinfo.value.code == 2


def test_preview_heating_with_land_cover_wms_mocked(
    synthetic_dem_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """preview --what heating --land-cover-wms calls fetch_lcm_for_dem.

    Monkeypatches the fetch to a synthetic LandCover so the test is hermetic
    (no real network) but exercises the full CLI plumbing path.
    """
    import numpy as np
    from rasterio.crs import CRS

    from thermal_model.io import DEM, LandCover

    captured: dict[str, object] = {}

    def fake_fetch(reference: DEM, **kwargs):  # type: ignore[no-untyped-def]
        captured["called"] = True
        captured["layer"] = kwargs.get("layer")
        captured["use_cache"] = kwargs.get("use_cache")
        # Return a uniform-bog LCM covering the DEM footprint.
        rows, cols = reference.shape
        return LandCover(
            classes=np.full((rows, cols), 11, dtype=np.int16),
            transform=reference.transform,
            crs=CRS.from_epsg(27700),
            cell_size_m=reference.cell_size_m,
        )

    monkeypatch.setattr(
        "thermal_model.cli.fetch_lcm_for_dem", fake_fetch, raising=False
    )
    # The CLI imports `fetch_lcm_for_dem` lazily from `thermal_model.io`
    # inside `_resolve_heating_args`; patch there instead.
    monkeypatch.setattr("thermal_model.io.fetch_lcm_for_dem", fake_fetch, raising=True)

    out = tmp_path / "heating_wms.png"
    rc = main(
        [
            "preview",
            "--dem",
            str(synthetic_dem_path),
            "--what",
            "heating",
            "--land-cover-wms",
            "--no-lcm-cache",
            "--datetime",
            "2026-06-21T12:00:00+01:00",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert captured.get("called") is True
    assert captured.get("layer") == "LC.10m.GB"
    assert captured.get("use_cache") is False
