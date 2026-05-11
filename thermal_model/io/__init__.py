"""Raster I/O. All GeoTIFF reads and writes go through this package."""

from thermal_model.io.dem import DEM, read_dem, write_raster_like
from thermal_model.io.land_cover import (
    DALES_LCM_ABSORPTIVITY,
    UKCEH_LCM_ABSORPTIVITY,
    LandCover,
    absorptivity_from_land_cover,
    read_land_cover,
)
from thermal_model.io.land_cover_wms import (
    LCM_WMS_DEFAULTS,
    UKCEH_LCM_PALETTE,
    fetch_lcm_for_dem,
)
from thermal_model.io.mosaic import mosaic_dems

__all__ = [
    "DALES_LCM_ABSORPTIVITY",
    "DEM",
    "LCM_WMS_DEFAULTS",
    "LandCover",
    "UKCEH_LCM_ABSORPTIVITY",
    "UKCEH_LCM_PALETTE",
    "absorptivity_from_land_cover",
    "fetch_lcm_for_dem",
    "mosaic_dems",
    "read_dem",
    "read_land_cover",
    "write_raster_like",
]
