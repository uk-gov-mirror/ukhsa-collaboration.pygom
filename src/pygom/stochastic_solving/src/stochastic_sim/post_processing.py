"""
Post processing of stochastic output
"""

import pandas as pd
import numpy as np

def bin_jumps(t, jumps, jump_names, target_time):
    """
    Given number of jumps between t_min and t_max
    """
    n_trans = jumps.shape[1]
    mid = 0.5 * (t[:-1] + t[1:])

    # empty matrix to receive scaled data = (new timepoints x n_trans)
    # minus one because we are looking at jumps which occur betwen timepoints
    # (e.g. 2 timepoints = 1 jump, 10 timepoints = 9)
    jump_bin = np.zeros((len(target_time)-1, n_trans))

    # if exact, each point corresponds to a transitions and has weight 1.
    for i in range(n_trans):
        hist, _ = np.histogram(mid, bins=target_time, weights=jumps[:,i])
        jump_bin[:, i] = hist

    df = pd.DataFrame({'t': target_time[:-1]})
    df[jump_names] = jump_bin
    
    return df

def extract_state_at_times(t, x, state_names, target_time):
    """
    Given time series of state values, get snapshots at target times.
    Appropriate for exact simulation outputs
    """
    # Find insertion positions
    idx = np.searchsorted(t, target_time, side="right") - 1

    # Clamp to range [0, len(t)-1]
    idx = np.clip(idx, 0, len(t)-1)

    df = pd.DataFrame({'t': target_time})
    df[state_names] = x[idx]
    
    return df

def interpolate_state_at_times(t, x, state_names, target_time):
    """
    Given time series of state values, get interpolated snapshots at target times.
    Appropriate for tau leap simulation outputs
    """

    n_state = x.shape[1]

    x_interp = np.zeros((len(target_time), n_state))  # empty matrix to receive scaled data = (new timepoints x vars)

    # linearly interpolate to new timepoints
    for i in range(n_state):
        x_interp[:, i]=np.interp(target_time, t, x[:,i])

    df = pd.DataFrame({'t': target_time})
    df[state_names] = x_interp
    
    return df