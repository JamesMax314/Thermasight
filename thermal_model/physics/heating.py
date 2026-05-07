"""Heating field — solar energy absorbed per unit area of terrain.

The heating field is the third leg of the Phase 2 model
(``docs/MODEL.md`` §2):

    H = I * alpha * s

where ``I`` is slope-projected clear-sky irradiance (W/m²),
``alpha`` is the surface's shortwave absorptivity (``1 - albedo``),
and ``s in {0, 1}`` is the cast-shadow mask.

The cast-shadow mask in this project attenuates the *beam* component
only — diffuse light comes from the whole sky and isn't blocked by
a single upwind ridge — so the practical assembly is

    H = alpha * (s * I_beam + I_diffuse)

with ``I_beam`` and ``I_diffuse`` from
:func:`thermal_model.solar.slope_irradiance` and ``s`` from
:func:`thermal_model.solar.cast_shadow_mask`.

Phase 2 ships a single scalar ``alpha`` default. Phase 4 will switch
in a per-cell ``alpha`` array driven by the UKCEH land cover lookup
in ``docs/DATA.md`` (the "wet ground is dead ground" effect — bog
absorptivity ~0.4 versus dry grass ~0.8 is the single biggest entry).
"""

from __future__ import annotations

import numpy as np

from thermal_model.solar.irradiance import SlopeIrradiance

#: Default shortwave absorptivity for upland Dales surfaces. Matches
#: the "dry grass / heather" entry in ``docs/DATA.md`` and is the
#: working default until Phase 4 land-cover support lands.
DEFAULT_ABSORPTIVITY: float = 0.80


def heating_field(
    irradiance: SlopeIrradiance,
    shadow_mask: np.ndarray,
    *,
    absorptivity: float | np.ndarray = DEFAULT_ABSORPTIVITY,
) -> np.ndarray:
    """Per-cell solar heating ``H`` in W/m².

    The cast shadow attenuates only the beam component:

        H = alpha * (s * I_beam + I_diffuse)

    Parameters
    ----------
    irradiance : SlopeIrradiance
        Per-cell beam and diffuse irradiance from
        :func:`thermal_model.solar.slope_irradiance`.
    shadow_mask : np.ndarray
        Cast-shadow mask from
        :func:`thermal_model.solar.cast_shadow_mask`. Same shape as
        ``irradiance.beam_wm2``. Values are ``1.0`` (sunlit), ``0.0``
        (in cast shadow), or ``NaN`` (invalid input cell). Soft
        (fractional) masks in ``[0, 1]`` are also accepted.
    absorptivity : float or np.ndarray, default 0.80
        Shortwave absorptivity ``alpha = 1 - albedo``, dimensionless
        on ``[0, 1]``. Scalar (uniform surface) or an array broadcasting
        with the irradiance arrays. Default is the dry-grass/heather
        upland-Dales value from ``docs/DATA.md``.

    Returns
    -------
    np.ndarray
        Float64 heating field in W/m², same shape as
        ``irradiance.beam_wm2``. NaN propagates from any of
        ``irradiance.beam_wm2``, ``irradiance.diffuse_wm2``,
        ``shadow_mask``, or an array ``absorptivity``.

    Raises
    ------
    ValueError
        If shapes disagree or if a scalar ``absorptivity`` lies
        outside ``[0, 1]``.
    """
    beam = irradiance.beam_wm2
    diffuse = irradiance.diffuse_wm2
    if beam.shape != diffuse.shape:
        raise ValueError(
            f"irradiance.beam_wm2 shape {beam.shape} does not match "
            f"irradiance.diffuse_wm2 shape {diffuse.shape}"
        )
    if shadow_mask.shape != beam.shape:
        raise ValueError(
            f"shadow_mask shape {shadow_mask.shape} does not match "
            f"irradiance shape {beam.shape}"
        )

    alpha: float | np.ndarray
    if isinstance(absorptivity, np.ndarray):
        if absorptivity.shape != beam.shape:
            raise ValueError(
                f"absorptivity shape {absorptivity.shape} does not match "
                f"irradiance shape {beam.shape}"
            )
        alpha = absorptivity.astype(np.float64, copy=False)
    else:
        alpha_scalar = float(absorptivity)
        if not 0.0 <= alpha_scalar <= 1.0:
            raise ValueError(f"absorptivity must be in [0, 1], got {alpha_scalar}")
        alpha = alpha_scalar

    out: np.ndarray = alpha * (shadow_mask * beam + diffuse)
    return out
