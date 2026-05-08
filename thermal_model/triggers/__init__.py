"""Trigger-point detection and export.

Phase 3 of the model. The trigger-potential raster from
:func:`thermal_model.physics.run_model` is a continuous field on
``[0, 1]``; this module turns the bright cells into discrete trigger
*points* suitable for plotting on a map or loading into XCTrack /
SeeYou / Google Earth as a KMZ.

Clustering uses connected components on a high-percentile mask
(``scipy.ndimage.label``); on a regular raster this is the natural
equivalent of DBSCAN with ``eps = cell_size`` and avoids the
``scikit-learn`` dependency (``CLAUDE.md`` §4).
"""

from thermal_model.triggers.cluster import TriggerPoint, cluster_triggers
from thermal_model.triggers.export import write_kmz

__all__ = ["TriggerPoint", "cluster_triggers", "write_kmz"]
