"""Coupling: heating × convergence → thermal-potential raster.

The Phase 2 model produces two independent fields:

* the heating field ``H`` (W/m²) from
  :func:`thermal_model.physics.heating_field` — where the air is
  warmed;
* the convergence field ``C`` (cell counts) from
  :func:`thermal_model.physics.flow_accumulation` on the inverted
  pit-filled DEM — where rising air would pool.

A useful thermal needs *both* a hot spot and a convergence feature
to release off, so the thermal-potential field is an "AND" of the
two. A naive product ``H · C`` over-weights ``C`` because the
dynamic range of flow accumulation (1 to ~10⁵) dwarfs that of
heating (0 to ~10³ W/m²); a single very-high-``C`` cell would
dominate. The geometric mean ``sqrt(H · C)`` from
``docs/MODEL.md`` §3 compresses the two onto a common scale before
combining, so cells strong on *one* axis but weak on the *other* get
pulled down and only cells that are strong on both float to the top.

Exposing the exponents

    P = H^p · C^q

lets the operator tune the balance. ``(p, q) = (0.5, 0.5)`` is the
geometric mean default. ``(0.7, 0.3)`` is heating-weighted, which
matches early-day conditions where SE-aspect dominates the
distinction (``CLAUDE.md`` §5). ``(0.3, 0.7)`` is
convergence-weighted, which matches afternoons when the whole
massif is warm and trigger geometry takes over. Phase 4 will
automate this time-of-day weighting; here it is just a knob.

The output is a *relative ranking* and not a physical quantity —
the units come out as ``(W/m²)^p · count^q``, which has no useful
interpretation in absolute terms. Plot on a percentile scale.
"""

from __future__ import annotations

import numpy as np


def thermal_potential(
    heating_wm2: np.ndarray,
    convergence: np.ndarray,
    *,
    heating_exponent: float = 0.5,
    convergence_exponent: float = 0.5,
) -> np.ndarray:
    """Couple heating and convergence into a thermal-potential field.

    Computes ``P = H^p · C^q`` per cell. The default
    ``(p, q) = (0.5, 0.5)`` is the geometric mean
    ``sqrt(H · C)`` recommended by ``docs/MODEL.md`` §3.

    Parameters
    ----------
    heating_wm2 : np.ndarray
        Heating field in W/m² (or any non-negative scalar field
        proportional to local heating). Same shape as ``convergence``.
    convergence : np.ndarray
        Convergence field — flow accumulation on the inverted
        pit-filled DEM, dimensionless cell counts. Same shape as
        ``heating_wm2``.
    heating_exponent : float, default 0.5
        Exponent ``p`` on the heating field. ``0`` removes the
        heating dependence entirely; ``1`` gives an unweighted
        product on this axis.
    convergence_exponent : float, default 0.5
        Exponent ``q`` on the convergence field. Same semantics as
        ``heating_exponent``.

    Returns
    -------
    np.ndarray
        Float64 thermal-potential array, same shape as the inputs.
        NaN propagates from either input. The output is a relative
        ranking, *not* a physical quantity — its units are
        ``(W/m²)^p · count^q``, which is meaningless in absolute
        terms. Display on a percentile scale.

    Raises
    ------
    ValueError
        If shapes disagree, if either exponent is negative (which
        would invert the ranking and is almost certainly not what
        the caller wants), or if either input contains a strictly
        negative value (NaN is not flagged).

    Notes
    -----
    Both inputs are expected to be non-negative; the formula
    ``x^p`` is otherwise complex-valued for non-integer ``p``. A
    cell with a true zero in either field becomes a true zero in
    the output, which is correct: no heat → no thermal, no
    convergence → no organised release.

    Cells where the convergence map shows ``0`` (e.g. boundary
    outlets in some flow-accumulation conventions) collapse the
    coupled potential to ``0`` regardless of heating, which is the
    right behaviour. To keep these cells visible add ``1`` to the
    convergence input before calling — the same compression already
    used by ``viz.plot_convergence``.
    """
    if heating_wm2.shape != convergence.shape:
        raise ValueError(
            f"heating_wm2 shape {heating_wm2.shape} does not match "
            f"convergence shape {convergence.shape}"
        )
    if heating_exponent < 0.0:
        raise ValueError(
            f"heating_exponent must be non-negative, got {heating_exponent}"
        )
    if convergence_exponent < 0.0:
        raise ValueError(
            f"convergence_exponent must be non-negative, got {convergence_exponent}"
        )

    # Treat NaN as "no information" rather than "less than zero" — the
    # comparison `< 0` is False for NaN, so NaN cells slip through and
    # propagate via the multiplication below.
    if np.any(heating_wm2 < 0.0):
        raise ValueError("heating_wm2 contains negative values")
    if np.any(convergence < 0.0):
        raise ValueError("convergence contains negative values")

    h = heating_wm2.astype(np.float64, copy=False)
    c = convergence.astype(np.float64, copy=False)

    # Skip the np.power call when an exponent is exactly 1 to avoid
    # 1**NaN-style oddities and a tiny amount of work; both `1.0` paths
    # are common (the heating- or convergence-only sensitivity sweep).
    if heating_exponent == 1.0:
        h_term: np.ndarray = h
    elif heating_exponent == 0.0:
        h_term = np.ones_like(h)
        # An exponent of 0 means "ignore this axis"; preserve NaN so
        # NaN-in-NaN-out still holds when the *other* axis is NaN.
        h_term = np.where(np.isnan(h), np.nan, h_term)
    else:
        h_term = np.power(h, heating_exponent)

    if convergence_exponent == 1.0:
        c_term: np.ndarray = c
    elif convergence_exponent == 0.0:
        c_term = np.ones_like(c)
        c_term = np.where(np.isnan(c), np.nan, c_term)
    else:
        c_term = np.power(c, convergence_exponent)

    out: np.ndarray = h_term * c_term
    return out
