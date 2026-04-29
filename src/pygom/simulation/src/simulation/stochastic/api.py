"""
Solver
"""

import numpy as np
import time
from .factory import make_stepper

# ------------
# Data classes
# ------------
# Result
# @dataclass
# class SimulationResult:
#     x: np.ndarray       # new state after timestep
#     t: float            # new time after timestep
#     jumps: np.ndarray   # number of times each jump occured between old and new timestep

# # Performace
# @dataclass
# class PerformanceMetrics:
#     wall_time_seconds: float
#     cpu_time_seconds: float | None = None

# # Output
# @dataclass
# class Output:
#     out: SimulationResult
#     config : SolverConfig
#     diag: SolverDiagnostics | None = None
#     performance: PerformanceMetrics | None = None

from .config import TauConfig, ExactConfig
from ..data_classes import Output, PerformanceMetrics, TimeSeriesResult, EventSeriesResult

# def sim_stochastic(
#     x0,
#     t_max,
#     ode_eqns,
#     state_change_mat,
#     x_min,
#     x_max,
#     *,
#     config,
#     t0=0.0,
#     proceed_if_rates_zero=False,
#     perf=False,
#     transition_mean_func=None,
#     transition_var_func=None
# ):

def sim_stochastic(
    x0,
    t_max,
    ode_eqns,
    state_change_mat,
    x_min,
    x_max,
    config,
    t0=0.0,
    proceed_if_rates_zero=False,
    perf=False,
    seed=None
    # **options
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
        x_min=x_min,
        x_max=x_max,
        proceed_if_rates_zero=proceed_if_rates_zero,
        seed=seed
        # **options
    )

    # Initial conditions
    t = t0
    x = np.array(x0, dtype=float)

    if perf:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

    # TODO: very similar. For now treat exact and tau leap steppers different

    if isinstance(config, ExactConfig):
        # Initialise output lists
        xs = [x.copy()]
        ts = [t]
        transition_idxs = []

        # Main loop
        while t < t_max:
            new_state = stepper.take_step(x, t)

            if new_state.end_sim:
                break

            x = new_state.x_new.copy()
            t = new_state.t_new

            xs.append(x)
            ts.append(t)
            transition_idxs.append(new_state.event_idx)

        x = np.array(xs)
        t = np.array(ts)
        transition_idxs = np.array(transition_idxs)

        result = EventSeriesResult(x=x, t=t, event_id=transition_idxs)

    elif isinstance(config, TauConfig):
        # Initialise output lists
        xs = [x.copy()]
        ts = [t]
        jumps = []

        # Main loop
        while t < t_max:
            new_state = stepper.take_step(x, t)

            if new_state.end_sim:
                break
            
            x = new_state.x_new.copy()
            t = new_state.t_new

            xs.append(x)
            ts.append(t)
            jumps.append(new_state.jumps)
        
        x = np.array(xs)
        t = np.array(ts)
        jumps = np.array(jumps)

        result = TimeSeriesResult(x=x, t=t, jumps=jumps)

    out = Output(result=result, config=config, diag=stepper.diag)

    if perf:
        wall_time_seconds = time.perf_counter() - start_time
        cpu_time_seconds = time.process_time() - start_cpu

        perf = {}
        perf['wall_time_seconds'] = wall_time_seconds
        perf['cpu_time_seconds'] = cpu_time_seconds

        out.performance = PerformanceMetrics(**perf)

    return out
