import time
import numpy as np

from .integrator import DeterministicSolver
from ..data_classes import Output, PerformanceMetrics, TimeSeriesResult

from ..data_classes import SolverConfig

from dataclasses import dataclass

@dataclass
class DeterministicConfig(SolverConfig):
    method = "Deterministic"

def sim_deterministic(
    x0,
    t,
    ode_eqns,
    trans_rates,
    n_state,
    n_trans,
    t0=0.0,
    perf=False,
    **kwargs
):
    """
    Solve deterministic system
    
    Parameters
    ----------
    x0 : numpy.ndarray
        Initial state values
    t_eval : float
        Maximum simulation time value
    ode_eqns : callable
        Transition rates as a function of state and time
    trans_rates : callable
        State change matrix (TODO: treat separately if constant, which is probably always the case)
    t0 : float : default = 0.0
        Initial time value
    perf : bool : default = False
        Include performance diagnostics in output if True

    Returns
    -------
    Output
    """

    stepper = DeterministicSolver(ode_eqns, trans_rates, n_state, n_trans)

    if perf:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

    result = stepper.integrate(t, x0, t0=t0, **kwargs)

    t = result.t
    x = result.x
    jumps = result.jumps
    diag = result.scipy_out
    # tmin = np.array(t[:-1])
    # tmax = np.array(t[1:])
    result = TimeSeriesResult(x=x, t=t, jumps=jumps)

    out = Output(result=result, config=DeterministicConfig(), diag=diag)

    if perf:
        wall_time_seconds = time.perf_counter() - start_time
        cpu_time_seconds = time.process_time() - start_cpu

        perf = {}
        perf['wall_time_seconds'] = wall_time_seconds
        perf['cpu_time_seconds'] = cpu_time_seconds

        out.performance = PerformanceMetrics(**perf)

    return out