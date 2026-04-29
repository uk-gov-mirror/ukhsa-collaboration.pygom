from .post_processing import *
from .stochastic.api import sim_stochastic
from .deterministic.api import sim_deterministic

from .data_classes import TimeSeriesResult, EventSeriesResult

import numpy as np


from .stochastic.config_api import build_config

def solve_stochastic(
    x0,
    t,
    ode_eqns,
    state_change_mat,
    x_min,
    x_max,
    n_trans,
    # *,
    config,
    t0=0.0,
    proceed_if_rates_zero=False,
    perf=False,
    seed=None
    # transition_mean_func=None,
    # transition_var_func=None,
):
    post_process = False

    if isinstance(t, np.ndarray):
        post_process = True
        t_max = t.max()
    elif isinstance(t, float):
        t_max = t
    else:
        raise ValueError("Invalid time format. Must be numpy.array or float.")

    result = sim_stochastic(
        x0,
        t_max,
        ode_eqns,
        state_change_mat,
        x_min,
        x_max,
        config=config,
        t0=t0,
        proceed_if_rates_zero=proceed_if_rates_zero,
        perf=perf,
        seed=seed)
    
    if post_process:
        if isinstance(result.result, TimeSeriesResult):
            result.result.x = interpolate_state_at_times(result.result.x, result.result.t, t)
            result.result.jumps = change_bins(result.result.jumps, result.result.t, t)
            result.result.t = t
        elif isinstance(result.result, EventSeriesResult):
            x = extract_state_at_target_times(result.result.x, result.result.t, t)
            jumps = bin_events(result.result.event_id, result.result.t[1:], t, n_trans) # t=0 is not an event
            result.result = TimeSeriesResult(t=t, x=x, jumps=jumps)

    return result


def solve_deterministic(
    x0,
    t,
    ode_eqns,
    trans_rates,
    n_state,
    n_trans,
    t0=0.0,
    perf=False,
    **kwargs):

    # # TODO: let user pass all of scipy integrates functionality

    # if not isinstance(t, np.ndarray):
    #     raise ValueError("Invalid time format. Must specify target times via numpy.array for deterministic.")

    result = sim_deterministic(
        x0=x0,
        t=t,
        ode_eqns=ode_eqns,
        trans_rates=trans_rates,
        n_state=n_state,
        n_trans=n_trans,
        perf=perf,
        t0=t0,
        **kwargs)
    
    return result
