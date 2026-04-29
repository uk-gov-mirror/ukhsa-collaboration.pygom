"""
Data classes for simulation outputs
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SolverDiagnostics:
    zero_rate_termination: bool = False

@dataclass
class SolverConfig:
    pass

@dataclass
class PerformanceMetrics:
    wall_time_seconds: float
    cpu_time_seconds: float | None = None

@dataclass
class SimulationResult:
    pass

@dataclass
class TimeSeriesResult(SimulationResult):
    t: np.ndarray       # 
    x: np.ndarray       # state at time, t
    jumps: np.ndarray   # number of times each jump occured between tmin and tmax

@dataclass
class EventSeriesResult(SimulationResult):
    t: np.ndarray
    x: np.ndarray
    event_id: np.ndarray   # 1 if jump occured and 0 if not.

@dataclass
class Output:
    result: SimulationResult
    config: SolverConfig
    diag: SolverDiagnostics | None = None
    performance: PerformanceMetrics | None = None
