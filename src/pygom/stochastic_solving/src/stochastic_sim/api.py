"""
Solver
"""

import numpy as np
import time
from dataclasses import dataclass

from .factory import make_stepper

from .core.base import SolverDiagnostics

from .config import SolverConfig

# ------------
# Data classes
# ------------
# Result
@dataclass
class SimulationResult:
    x: np.ndarray       # new state after timestep
    t: float            # new time after timestep
    jumps: np.ndarray   # number of times each jump occured between old and new timestep

# Performace
@dataclass
class PerformanceMetrics:
    wall_time_seconds: float
    cpu_time_seconds: float | None = None

# Output
@dataclass
class Output:
    out: SimulationResult
    config : SolverConfig
    diag: SolverDiagnostics | None = None
    performance: PerformanceMetrics | None = None


def solve(
    x0,
    t_max,
    ode_eqns,
    state_change_mat,
    x_min,
    x_max,
    *,
    config,
    t0=0.0,
    proceed_if_rates_zero=True,
    perf=False,
    transition_mean_func=None,
    transition_var_func=None
):
    """
    Solve stochastic system
    
    Parameters
    ----------
    x0 : numpy.ndarray
        Initial state values
    t_max : float
        Maximum simulation time value
    ode_eqns : callable
        Transition rates as a function of state and time
    state_change_mat : callable
        State change matrix (TODO: treat separately if constant, which is probably always the case)
    x_min : numpy.ndarray
        Minimum allowable values for each state variable.
    x_max : numpy.ndarray
        Maximum allowable values for each state variable.
    config : SolverConfig
        What kind of stepper and any relevant parameters
    t0 : float : default = 0.0
        Initial time value
    proceed_if_rates_zero : bool : default = True
        If True, continue with simulation when reaction rates are all zero. Otherwise terminate.
    perf : bool : default = False
        Include performance diagnostics in output if True

    Returns
    -------
    Output
    """
    stepper = make_stepper(
        config=config,
        transition_func=ode_eqns,
        state_change_mat=state_change_mat,
        transition_mean_func=transition_mean_func,
        transition_var_func=transition_var_func,
        x_min=x_min,
        x_max=x_max,
        proceed_if_rates_zero=proceed_if_rates_zero,
    )

    # Initial conditions
    t = t0
    x = np.array(x0, dtype=float)

    # Initialise output lists
    xs = [x.copy()]
    ts = [t]
    jumps = []

    if perf:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

    # Main loop
    while t < t_max:
        new_state = stepper.take_step(x, t)

        t = new_state.t_new
        x = new_state.x_new

        if new_state.end_sim:
            break

        xs.append(x.copy())
        ts.append(t)
        jumps.append(new_state.jumps)


    result = SimulationResult(x=np.array(xs), t=np.array(ts), jumps=np.array(jumps))

    out = Output(out=result, config=config, diag=stepper.diag)

    if perf:
        wall_time_seconds = time.perf_counter() - start_time
        cpu_time_seconds = time.process_time() - start_cpu

        perf = {}
        perf['wall_time_seconds'] = wall_time_seconds
        perf['cpu_time_seconds'] = cpu_time_seconds

        out.performance = PerformanceMetrics(**perf)

    return out