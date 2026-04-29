"""
Tau Leap
"""

from dataclasses import dataclass
import numpy as np

from ..base import StochasticLeap, TimeStep
from ....data_classes import SolverDiagnostics
from .precaution import TauRefiner
from .method import TauMethod
from ..step_checker import JumpChecker

@dataclass
class TauLeapDiagnostics(SolverDiagnostics):
    n_failed_step: int = 0

class TauLeap(StochasticLeap):
    """
    Base class for tau leap algorithms. Building on `StochasticLeap`.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.


    """
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero,
                 retry_max,
                 tau_rescale,
                 tau_method : TauMethod,
                 proposal_checker : JumpChecker,
                 tau_refiner : TauRefiner,
                 seed=None):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, seed)

        self.tau_method = tau_method
        self.proposal_checker = proposal_checker
        self.tau_refiner = tau_refiner
        self.retry_max = retry_max
        self.tau_rescale = tau_rescale
        self.diag = TauLeapDiagnostics()

    def _propose_jump(self, rates, tau):
        # jumps = np.random.poisson(rates * tau)
        jumps = self.rng.poisson(rates * tau)
        return jumps, tau
    
    def _modify_tau(self, tau):
        return tau * self.tau_rescale

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

    def _zero_rate_behavior(self, x, t, tau):
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
            return TimeStep(x_new=x, t_new=t+tau, jumps=np.zeros(self.n_reactions), end_sim=False)
        else:
            self.diag.zero_rate_termination=True
            return TimeStep(x_new=None, t_new=None, jumps=None, end_sim=True)

    def take_step(self, x, t):
        """
        Called by user. Returns (x_new, t_new).
        """
        rates, changes = self._compute_rates_and_changes(x, t)

        thresholds = self._jump_thresholds(x, changes)              # TODO: might not always be needed
        tau, success = self.tau_method.compute_tau(x, t, rates, thresholds)

        if success==False:
            return TimeStep(x_new=None, t_new=None, jumps=None, end_sim=True)

        tau = self.tau_refiner.refine_tau(tau, thresholds, rates)

        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t, tau)

        for _ in range(self.retry_max + 1):
            # Propose jump
            jumps, dt = self._propose_jump(rates, tau)

            # Calculate new state
            x_proposal = self._get_new_x(x, changes, jumps)

            step_proposal = TimeStep(x_new=x_proposal, t_new=t+dt, jumps=jumps, end_sim=False)

            # Check if jump is legal
            if self.proposal_checker.is_legal(step_proposal, self.x_min, self.x_max, thresholds):
                return step_proposal
            else:
                self.diag.n_failed_step += 1
                tau = self._modify_tau(tau)
        raise RuntimeError(f"Forbidden values still encountered after {self.retry_max} attempts")