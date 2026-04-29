from unittest import main, TestCase

import numpy as np
import scipy.optimize

from pygom import SquareLoss, NormalLoss
from pygom.model import common_models


class TestModelEstimate(TestCase):

    def setUp(self):
        # define the model and parameters
        beta = 0.5
        gamma = 1.0/3.0
        self.ode = common_models.SIR_norm({'beta': beta, 'gamma': gamma})

        # the initial state, normalized to zero one
        i0 = 1.27e-6
        self.x0 = [1-i0, i0, 0]
        # set the time sequence that we would like to observe
        self.t = np.linspace(0, 150, 100)
        self.ode.initial_values = (self.x0, self.t[0])
        # Standard.  Find the solution.
        # self.solution = self.ode.integrate(self.t[1::])
        self.solution = self.ode.solve_deterministic(self.t).result.x

        # initial value
        self.theta = np.array([0.4, 0.2])
        # what the estimates should be close to
        self.target = np.array([beta, gamma])

        # constraints
        EPSILON = np.sqrt(np.finfo(float).eps)

        self.box_bounds = [(EPSILON, 5), (EPSILON, 5)]

    def test_single_state_func(self):
        """
        Just to see if the functions manage to run at all
        """
        y = self.solution[1::, 2]
        # test out whether the single state function 'ok'
        sir_obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],
                             self.t[1::], y, 'R')
        sir_obj.cost()
        sir_obj.gradient()
        sir_obj.hessian()

    def test_SIR_Estimate_SquareLoss(self):
        y = self.solution[1::, 1:3]
        sir_obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0], self.t[1::],
                             y, ['I', 'R'])

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                         jac=sir_obj.sensitivity,
                                         x0=self.theta,
                                         method='SLSQP',
                                         bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )


    def test_SIR_Estimate_SquareLoss_Adjoint(self):
        y = self.solution[1::, 1:3]

        sir_obj = SquareLoss(self.theta, self.ode, self.x0, self.t[0],
                             self.t[1::], y, ['I', 'R'])

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                         jac=sir_obj.adjoint,
                                         x0=self.theta,
                                         method='SLSQP',
                                         bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )

    def test_SIR_Estimate_NormalLoss(self):
        y = self.solution[1::, 1:3]

        sir_obj = NormalLoss(self.theta, self.ode, self.x0, self.t[0],
                             self.t[1::], y, ['I', 'R'])

        res_QP = scipy.optimize.minimize(fun=sir_obj.cost,
                                         jac=sir_obj.sensitivity,
                                         x0=self.theta,
                                         method='SLSQP',
                                         bounds=self.box_bounds)

        self.assertTrue(
            np.allclose(res_QP['x'], self.target, rtol=1e-2, atol=1e-2),
            msg=f"Values differ:\nest={res_QP['x']}\ntarget={self.target}\ndiff={res_QP['x'] - self.target}"
        )

    def tearDown(self):
        self.ode = None
        self.solution = None
        self.x0 = None
        self.t = None
        self.theta = None
        self.target = None


if __name__ == '__main__':
    main()
