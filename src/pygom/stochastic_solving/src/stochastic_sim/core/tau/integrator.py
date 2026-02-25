"""
Tau Leap
"""

from dataclasses import dataclass
import numpy as np

from ..base import SolverDiagnostics, StochasticLeap
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
                 tau_refiner : TauRefiner):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero)

        self.tau_method = tau_method
        self.proposal_checker = proposal_checker
        self.tau_refiner = tau_refiner
        self.retry_max = retry_max
        self.tau_rescale = tau_rescale
        self.diag = TauLeapDiagnostics()

    def _propose_jump(self, rates, tau):
        jumps = np.random.poisson(rates * tau)
        return jumps, tau
    
    def _modify_tau(self, tau):
        return tau * self.tau_rescale

    def take_step(self, x, t):
        """
        Called by user. Returns (x_new, t_new).
        """
        rates, changes = self._compute_rates_and_changes(x, t)
        
        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t)

        thresholds = self._jump_thresholds(x, changes)              # TODO: might not always be needed
        tau = self.tau_method.compute_tau(x, t, rates, thresholds)
        tau = self.tau_refiner.refine_tau(tau, thresholds, rates)

        for _ in range(self.retry_max + 1):
            # Propose jump
            jumps, dt = self._propose_jump(rates, tau)

            # Calculate new state
            x_proposal = self._get_new_x(x, changes, jumps)

            step_proposal = self._package_output(x_new=x_proposal, t_new=t+dt, jumps=jumps, end_sim=False)

            # Check if jump is legal
            if self.proposal_checker.is_legal(step_proposal, self.x_min, self.x_max, thresholds):
                return step_proposal
            else:
                self.diag.n_failed_step += 1
                tau = self._modify_tau(tau)
        raise RuntimeError(f"Forbidden values still encountered after {self.retry_max} attempts")