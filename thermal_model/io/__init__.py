"""Raster I/O. All GeoTIFF reads and writes go through this package."""

from thermal_model.io.dem import DEM, read_dem, write_raster_like

__all__ = ["DEM", "read_dem", "write_raster_like"]
