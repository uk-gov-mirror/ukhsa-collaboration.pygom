"""
Tau leap step size method
"""

from abc import ABC, abstractmethod
import numpy as np
import warnings

class TauMethod(ABC):
    def __init__(self, transition_func, state_change_mat):
        self.transition_func = transition_func
        self.state_change_mat = state_change_mat

    @abstractmethod
    def compute_tau(self, x, t, rates):
        pass

class Fixed(TauMethod):
    def __init__(self, transition_func, state_change_mat, tau):
        super().__init__(transition_func, state_change_mat)
        self.tau = tau

    def compute_tau(self, x, t, rates, thresholds):
        return(self.tau, True)

class Adaptive(TauMethod):
    def __init__(self, transition_func, state_change_mat, transition_mean_func, transition_var_func):
        super().__init__(transition_func, state_change_mat)

        self.transition_mean_func = transition_mean_func
        self.transition_var_func = transition_var_func

    def _get_min_changes(self, x, t, thresholds):
        """
        Get the minimum amounts by which the transition rate (propensity) functions, a(x), can change
        by considering the action of each solo transition.

        Output:
            np.ndarray delta
            where delta[i] = min_j |a_i(x + v_j) - a_i(x)|
        """
        changes = self.state_change_mat(x, t)

        # Current propensity functions, a_i(x)
        rates_before = self.transition_func(x, t)

        # Rule out reactions which cannot possibly without generating illegal states
        noncritical_idx = np.where(thresholds!=0)[0]

        # Holder for rates after each transition occurring
        # TODO: n_p should be in base class
        n_reactions = len(noncritical_idx)
        if n_reactions == 0:
            raise RuntimeError("Simulation cannot proceed: No reactions can occur without generating invalid states.")
        n_propensities = rates_before.size
        rates_after = np.empty((n_reactions, n_propensities))

        # Get new propensity functions after each reaction occurs, a_i(x + v_j)
        x_new = x[:, None] + changes[:, noncritical_idx ]
        for i in range(n_reactions):
            rates_after[i] = self.transition_func(x_new[:, i ], t)

        # Find changes per propensity function per reaction, delta_ij = |a_i(x + v_j) - a_i(x)|
        delta = np.abs(rates_after - rates_before)

        # Find mininum non zero changes per propensity function
        masked = np.where(delta == 0, np.inf, delta)
        delta_min = np.min(masked, axis=0)
        delta_min[delta_min == np.inf] = 0  # cleanup if zero is the only option

        return delta_min

class Cao2006(Adaptive):
    """
    Functions to compute step size according to Cao 2006 paper
    Open pdf version: https://people.cs.vt.edu/~ycao/publication/adaptivetau.pdf
    DOI: https://doi.org/10.1063/1.2745299
    """
    def __init__(self, transition_func, state_change_mat, transition_mean_func, transition_var_func, epsilon):
        super().__init__(transition_func, state_change_mat, transition_mean_func, transition_var_func)
        self.epsilon = epsilon

    def compute_tau(self, x, t, rates, thresholds):
        # 8a and 8b
        mu = self.transition_mean_func(x, t)
        sigma2 = self.transition_var_func(x, t)

        # changes must be smaller than the bound
        delta_min = self._get_min_changes(x, t, thresholds)
        bound = np.maximum(self.epsilon*rates, delta_min)

        # timesteps which satisfy bound constraint (ignoring cases where bound = 0)
        valid = bound > 0
        # term1 = np.where((mu != 0) & valid, bound / np.abs(mu), np.inf)
        # term2 = np.where((sigma2 != 0) & valid, bound**2 / sigma2, np.inf)

        term1 = np.full_like(mu, np.inf, dtype=float)
        np.divide(
            bound,
            np.abs(mu),
            out=term1,
            where=(mu != 0) & valid
        )

        term2 = np.full_like(sigma2, np.inf, dtype=float)
        np.divide(
            bound**2,
            sigma2,
            out=term2,
            where = (sigma2 != 0) & valid
        )

        # timestep is the minimum one available
        tau = min(np.min(term1), np.min(term2))

        if not np.isfinite(tau):
            warnings.warn(f"Tau selection failed: Infinite step size calculated at t = {t}")
            return tau, False
        if tau == 0:
            warnings.warn(f"Tau selection failed: Zero step size calculated at t = {t}")
            return tau, False
        if tau < 0:
            warnings.warn(f"Tau selection failed: Negative step size calculated at t = {t}")
            return tau, False

        return tau, True


    # def compute_tau(self, x, t, rates, thresholds):
    #     mu = self.transition_mean_func(x, t)
    #     sigma2 = self.transition_var_func(x, t)

    #     # Bound depends on the sum over all rates
    #     a0 = rates.sum()
    #     bound = self.epsilon * a0

    #     return min(
    #         bound / np.abs(mu),
    #         bound**2 / sigma2
    #         )
