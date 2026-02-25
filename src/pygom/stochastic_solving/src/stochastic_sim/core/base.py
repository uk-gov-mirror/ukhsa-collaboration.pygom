# import numpy as np
# from abc import ABC, abstractmethod
# import logging

# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import poisson

# import time

# from dataclasses import dataclass


"""
Stochastic jumper base class
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# ------------
# Data classes
# ------------
# Step proposal
@dataclass
class Step:
    x_new: np.ndarray       # new state after timestep
    t_new: float            # new time after timestep
    jumps: np.ndarray       # number of times each jump occured between old and new timestep
    end_sim: bool           # True if simulation is to be prematurely ended (e.g if rates
                            # are all = 0 and user has decided these are grounds to abort)

# Solver diagnostics
@dataclass
class SolverDiagnostics:
    zero_rate_termination: bool = False

class StochasticLeap(ABC):
    """
    Base class for stocahstic stepper algorithms

    Parameters
    ----------
    transition_func : callable
        Returns vector of reaction rates
    state_change_mat : callable
        Returns the stoichiometric matrix
    x_min : numpy.ndarray
        Minimum allowable values for each state variable.
    x_max : numpy.ndarray
        Maximum allowable values for each state variable.
    proceed_if_rates_zero : bool
        If True, continue with simulation when reaction rates are all zero. Otherwise terminate.
    default_dt : float : default = 1
        Timestep used when rates are zero, but algorithm is required to proceed.

    Methods
    -------
    take_step()
        Execute the optimization procedure.
    """

    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, default_dt = 1):
        self.transition_func = transition_func
        self.state_change_mat = state_change_mat
        self.proceed_if_rates_zero = proceed_if_rates_zero
        self.x_min = x_min
        self.x_max = x_max
        self.default_dt = default_dt

    def _compute_rates_and_changes(self, x, t):
        """
        For the current timestep, calculate reaction rates and state change matrix.

        Parameters
        ----------
        x : numpy.ndarray
            Current state vector.
        t : float
            Current time.

        Returns
        -------
        rates : numpy.ndarray
            reaction rates
        changes : numpy.ndarray
            state change matrix
        """
        rates = self.transition_func(x, t)
        self.n_reactions = len(rates)

        if np.any(rates < 0):
            raise RuntimeError("Negative reaction rates encountered")
    
        changes = self.state_change_mat(x, t)

        return rates, changes

    def _jump_thresholds(self, x, changes):
        """
        Calculate the maximum number of times each individual reaction can occur
        before an illegal jump is made.

        Parameters
        ----------
        x : numpy.ndarray
            Current state vector.
        changes : numpy.ndarray
            State-change matrix specifying how each reaction modifies the state.

        Returns
        -------
        numpy.ndarray
            The maximum number of times each reaction can occur without violating
            state constraints.
        """

        # Difference between current state and state limits
        min_margin = x - self.x_min
        max_margin = self.x_max - x

        # Margins need to be broadcast across all reactions
        min_margin = min_margin[:, None]
        max_margin = max_margin[:, None]

        thresholds_min = np.full_like(changes, np.inf, dtype=float)
        thresholds_max = np.full_like(changes, np.inf, dtype=float)

        np.divide(
            min_margin,
            -changes,
            out=thresholds_min,
            where=changes < 0
        )

        np.divide(
            max_margin,
            changes,
            out=thresholds_max,
            where=changes > 0
        )

        thresholds_min = np.floor(thresholds_min)
        thresholds_max = np.floor(thresholds_max)

        return np.min(np.minimum(thresholds_min, thresholds_max), axis=0)

    def _get_new_x(self, x, changes, jumps):
        """
        Calculate the new state

        Parameters
        ----------
        x : numpy.ndarray
            Current state vector
        changes : numpy.ndarray
            State-change matrix specifying how each reaction modifies the state.
        jumps : numpt.ndarray
            Number of times each reaction occurs in the current timestep

        Returns
        -------
        numpy.ndarray
            The new state vector
        """
        
        return x + changes @ jumps

    @abstractmethod
    def _propose_jump(self, x, t, rates, *args):
        """
        Return (jumps, dt).
        """
        pass

    def _zero_rate_behavior(self, x, t):
        """
        Decide what to do when reaction rates are all zero.
        If rates are guaranteed to remain at zero, we might wish to abort the simulation to save time.
        If rates might possibly become non zero later on, we continue.

        Parameters
        ----------
        x : numpy.ndarray
            Current state vector.
        t : float
            Current time.

        Returns
        -------
        Step
            Step.x_new = new state values (which are unchanged from old ones)
            Step.t_new = new time
            Step.jumps = reaction occurances
            Step.end_sim = True if all rates = 0 led to simulation termination
        """
        if self.proceed_if_rates_zero:
            return self._package_output(x_new=x, t_new=t+self.default_dt, jumps=np.zeros(self.n_reactions), end_sim=False)
        else:
            self.diag.zero_rate_termination=False
            return self._package_output(x_new=x, t_new=t, jumps=np.zeros(self.n_reactions), end_sim=True)

    @abstractmethod
    def take_step(self, x, t):
        """
        Move system one timestep from (x, t) to (x_new, t_new)
        """
        pass

    def _package_output(self, x_new, t_new, jumps, end_sim):
        return Step(x_new=x_new, t_new=t_new, jumps=jumps, end_sim=end_sim)