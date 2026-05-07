"""Solar position, clear-sky irradiance, and slope-projected irradiance.

Phase 2 of the model (see ``docs/MODEL.md`` §2 and ``docs/ROADMAP.md``).
The cast-shadow mask and full heating field are added alongside this
module in subsequent steps.
"""

from thermal_model.solar.irradiance import (
    ClearSkyIrradiance,
    SlopeIrradiance,
    clear_sky_irradiance,
    slope_irradiance,
)
from thermal_model.solar.position import SolarPosition, solar_position
from thermal_model.solar.shadow import cast_shadow_mask

__all__ = [
    "ClearSkyIrradiance",
    "SlopeIrradiance",
    "SolarPosition",
    "cast_shadow_mask",
    "clear_sky_irradiance",
    "slope_irradiance",
    "solar_position",
]
