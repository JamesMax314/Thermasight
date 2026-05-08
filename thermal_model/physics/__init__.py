"""Physics: the inverted-treacle engine, heating, drift, and trigger logic."""

from thermal_model.physics.coupling import thermal_potential
from thermal_model.physics.flow import dinf_flow_directions, flow_accumulation
from thermal_model.physics.heating import DEFAULT_ABSORPTIVITY, heating_field
from thermal_model.physics.hydrology import fill_pits, resolve_flats
from thermal_model.physics.pipeline import RunResult, run_model
from thermal_model.physics.wind_tilt import wind_tilt_ramp

__all__ = [
    "DEFAULT_ABSORPTIVITY",
    "RunResult",
    "dinf_flow_directions",
    "fill_pits",
    "flow_accumulation",
    "heating_field",
    "resolve_flats",
    "run_model",
    "thermal_potential",
    "wind_tilt_ramp",
]
