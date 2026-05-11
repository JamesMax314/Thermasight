"""Visualization helpers: hillshade and diagnostic matplotlib plotters.

Plotting code lives here and is never imported back into ``physics`` or
``terrain``. Anything in this package is a *diagnostic*; the project's
deliverables are GeoTIFF and KMZ exports, not figures.

The Phase 1 :func:`hillshade` is a Lambertian shading function for
visualization only. The Phase 2 solar pipeline will introduce a
separate, physical hillshade (with cast shadows from a horizon scan)
under ``thermal_model.solar``; the two are intentionally distinct.
"""

from thermal_model.viz.diagnostics import (
    plot_absorptivity,
    plot_aspect,
    plot_convergence,
    plot_cycle_period,
    plot_draft_potential,
    plot_heating,
    plot_land_cover,
    plot_leak,
    plot_overlay,
    plot_profile_curvature,
    plot_slope,
    plot_trigger_potential,
    plot_weighted_convergence,
)
from thermal_model.viz.hillshade import hillshade

__all__ = [
    "hillshade",
    "plot_absorptivity",
    "plot_aspect",
    "plot_convergence",
    "plot_cycle_period",
    "plot_draft_potential",
    "plot_heating",
    "plot_land_cover",
    "plot_leak",
    "plot_overlay",
    "plot_profile_curvature",
    "plot_slope",
    "plot_trigger_potential",
    "plot_weighted_convergence",
]
