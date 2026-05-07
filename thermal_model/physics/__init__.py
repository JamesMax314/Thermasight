"""Physics: the inverted-treacle engine, heating, drift, and trigger logic."""

from thermal_model.physics.flow import dinf_flow_directions, flow_accumulation
from thermal_model.physics.hydrology import fill_pits

__all__ = ["dinf_flow_directions", "fill_pits", "flow_accumulation"]
