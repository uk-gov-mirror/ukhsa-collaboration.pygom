import numpy as np
from scipy.integrate import solve_ivp
from typing import Any
from dataclasses import dataclass

# ------------------
# Solver diagnostics
# ------------------

# TODO: put dataclass in other file?

@dataclass
class DeterministicOutput:
    t: np.ndarray
    x: np.ndarray
    jumps: np.ndarray
    scipy_out: Any

class DeterministicSolver:
    def __init__(self, ode, trans_rates, n_state, n_trans):
        self.ode = ode
        self.trans_rates = trans_rates
        self.n_state = n_state
        self.n_trans = n_trans

    def augmented_ode(self, t, x_aug):
        """
        Augment state vector with transition rates
        x_aug = [ [states], [transitions] ]

        Avoid building new compartments explicitly:
        - checks already done
        - initial conditions taken care of
        """
        x = x_aug[:self.n_state]
        
        # change in states
        dx = self.ode(x, t)
        
        # transition occurances
        dT = self.trans_rates(x, t)
        
        return np.concatenate([dx, dT])
    
    def augmented_jacobian(self, t, x_aug):
        """

        """

        # TODO: be explicit with sparsity
        # assert jac_x.shape == (self.n_state, self.n_state)
        # assert jac_t.shape[1] == self.n_state
        # assert jac_t.shape[0] == len(x_aug) - self.n_stat

        x = x_aug[:self.n_state]
        
        # change in states
        jac_x = self.jacobian(x, t)
        
        # transition occurances
        jac_t = self.rates_jacobian(x, t)

        jac_aug = np.zeros( (len(x_aug), len(x_aug)) )

        jac_aug[:self.n_state, :self.n_state] = jac_x
        jac_aug[self.n_state:, :self.n_state] = jac_t
        
        return jac_aug

    def integrate(self, t, x0, t0=0.0, **kwargs):

        trans0 = np.zeros(self.n_trans)
        x0_aug = np.concatenate([x0, trans0])

        if isinstance(t, float):
            sol = solve_ivp(
                fun = self.augmented_ode,
                t_span = (t0, t),
                y0 = x0_aug,
                jac = self.augmented_jacobian,
                **kwargs
            )
        elif isinstance(t, np.ndarray):
            sol = solve_ivp(
                fun = self.augmented_ode,
                t_span = (t0, t[-1]),
                y0 = x0_aug,
                t_eval=t,
                jac = self.augmented_jacobian,
                **kwargs
            )
        else:
            raise ValueError("Invalid time format. Must be numpy.array or float.")

        # TODO: popping seems not approved due to immutable object. Pop the things we want
        # Everything remaining is categorised as "scipy diagnostic info"
        t = sol.t
        y = sol.y

        states = y[:self.n_state, :]
        trans = y[self.n_state:, :]
        dT = np.diff(trans, axis=1)

        return DeterministicOutput(
            t=t,
            x=states.T,
            jumps=dT.T,
            scipy_out=sol
            )