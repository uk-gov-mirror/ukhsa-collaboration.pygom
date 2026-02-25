"""
Exact methods
"""

from dataclasses import dataclass
import numpy as np

from ..base import SolverDiagnostics, StochasticLeap

# No exact method diagnostic outputs
@dataclass
class DirectDiagnostics(SolverDiagnostics):
    pass
@dataclass
class FirstReactionDiagnostics(SolverDiagnostics):
    pass

# ============================================================
# First Reaction Method
# ============================================================
class FirstReaction(StochasticLeap):
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, default_dt = 1):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, default_dt = default_dt)
        self.diag = DirectDiagnostics()

    def _propose_jump(self, rates):
        jump_times = np.random.exponential(1.0 / rates)

        # find reaction with smallest time
        k = np.argmin(jump_times)

        jumps = np.zeros_like(rates, dtype=int)
        jumps[k] = 1
        dt = jump_times[k]

        return jumps, dt
    
    def take_step(self, x, t):
        rates, changes = self._compute_rates_and_changes(x, t)

        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t)

        jumps, dt = self._propose_jump(rates)
        x_new = self._get_new_x(x, changes, jumps)
        
        return self._package_output(x_new=x_new, t_new=t+dt, jumps=jumps, end_sim=False)

# ============================================================
# Direct Reaction Method
# ============================================================
class DirectReaction(StochasticLeap):
    def __init__(self, transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, default_dt = 1):
        super().__init__(transition_func, state_change_mat, x_min, x_max, proceed_if_rates_zero, default_dt = default_dt)
        self.diag = FirstReactionDiagnostics()

    def _propose_jump(self, rates):
        total_rate = rates.sum()
        transition_index = np.random.choice(len(rates), p=rates / total_rate)

        jumps = np.zeros(len(rates), dtype=np.int8)
        jumps[transition_index] = 1

        dt = np.random.exponential(1.0 / total_rate)

        return jumps, dt

    def take_step(self, x, t):
        rates, changes = self._compute_rates_and_changes(x, t)

        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t)

        jumps, dt = self._propose_jump(rates)
        x_new = self._get_new_x(x, changes, jumps)
        
        return self._package_output(x_new=x_new, t_new=t+dt, jumps=jumps, end_sim=False)