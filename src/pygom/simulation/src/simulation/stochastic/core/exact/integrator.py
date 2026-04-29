"""
Exact methods
"""

from dataclasses import dataclass
import numpy as np

from ..base import StochasticLeap, EventStep
from ....data_classes import SolverDiagnostics

# No exact method diagnostic outputs
@dataclass
class DirectDiagnostics(SolverDiagnostics):
    pass

@dataclass
class FirstReactionDiagnostics(SolverDiagnostics):
    pass

class ExactLeap(StochasticLeap):
    """
    Base class for tau leap algorithms. Building on `StochasticLeap`.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.


    """
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed=None):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed)
        self.diag = DirectDiagnostics()
    
    def _get_new_x(self, x, changes, transition_id):
        """
        TODO: docstring

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
        return x + changes[:, transition_id]


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
        self.diag.zero_rate_termination=True
        return EventStep(x_new=None, t_new=None, event_idx=None, end_sim=True)

# ============================================================
# First Reaction Method
# ============================================================
class FirstReaction(ExactLeap):
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed=None):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed)
        self.diag = DirectDiagnostics()

    def _propose_jump(self, rates):
        # jump_times = np.random.exponential(1.0 / rates)
        jump_times = self.rng.exponential(1.0 / rates)

        # find reaction with smallest time
        transition_idx = np.argmin(jump_times)

        # jumps = np.zeros_like(rates, dtype=int)
        # jumps[transition_id] = 1
        dt = jump_times[transition_idx]

        return transition_idx, dt
    
    def take_step(self, x, t):
        rates, changes = self._compute_rates_and_changes(x, t)

        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t)

        transition_idx, dt = self._propose_jump(rates)
        x_new = self._get_new_x(x, changes, transition_idx)
        
        return EventStep(x_new=x_new, t_new=t+dt, event_idx=transition_idx, end_sim=False)

# ============================================================
# Direct Reaction Method
# ============================================================
class DirectReaction(ExactLeap):
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed=None):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed)
        self.diag = FirstReactionDiagnostics()

    def _propose_jump(self, rates):
        total_rate = rates.sum()
        # transition_idx = np.random.choice(len(rates), p=rates / total_rate)
        transition_idx = self.rng.choice(len(rates), p=rates / total_rate)

        # jumps = np.zeros(len(rates), dtype=np.int8)
        # jumps[transition_index] = 1

        # dt = np.random.exponential(1.0 / total_rate)
        dt = self.rng.exponential(1.0 / total_rate)

        return transition_idx, dt

    def take_step(self, x, t):
        rates, changes = self._compute_rates_and_changes(x, t)

        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t)

        transition_idx, dt = self._propose_jump(rates)
        x_new = self._get_new_x(x, changes, transition_idx)
        
        return EventStep(x_new=x_new, t_new=t+dt, event_idx=transition_idx, end_sim=False)