"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOde']
import logging
import warnings

import copy
from numbers import Number

import numpy as np
import sympy
#import sympy.matrices.matrices
import scipy.stats

# from dask.distributed import Client
from .parallel_backends import run_dask, run_multiprocessing

from . import ode_utils
from .deterministic import DeterministicOde
from .stochastic_simulation import cle, exact, firstReaction, tauLeap, hybrid
from .transition import TransitionType, Transition
from ._model_errors import InputError, SimulationError
from ._model_verification import checkEquation, simplifyEquation
from . import _ode_composition

from .maths import (StateChangeMatrix, 
                    TransitionMean, 
                    TransitionVariance,
                    TransitionJacobian)

from ..simulation.src.simulation.api import solve_stochastic, solve_deterministic

from ..simulation.src.simulation.stochastic.config_api import build_config


from .ode_utils.variable_store import CallableParameter

class SimulateOde(DeterministicOde):
    '''
    This builds on top of :class:`DeterministicOde` which we
    simulate the outcome instead of solving it deterministically

    Parameters
    ----------
    state: list
        A list of states (string) or (string, (numeric, numeric)) if specifying limits
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string,string))
    transition: list
        A list of transition (:class:`Transition`) #TODO Now this might actually only be deterministic ODE objects. Check.
    transition: list
        A list of events (:class:`Event`)
    birth_death: list                              #TODO Now these are wrapped in events. Should try to make work for back compat.
        A list of birth or death process (:class:`Transition`)
    ode: list
        A list of ode (:class:`Transition`)

    '''
    # The compiled math functions within this class
    _maths_methods = DeterministicOde._maths_methods + [
        StateChangeMatrix,
        TransitionMean,
        TransitionVariance,
        TransitionJacobian
    ]

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 event=None,
                 birth_death=None,
                 ode=None,
                 backend='lambda'
                 ):
        '''
        Constructor that is built on top of DeterministicOde
        '''
        logging.debug(self._maths_methods)
        super(SimulateOde, self).__init__(state,
                                          param,
                                          derived_param,
                                          transition,
                                          event,
                                          birth_death,
                                          ode,
                                          backend)

        # self.pre_tau=None       # If tau is set, then this overrides the adaptive tau leap.
        # self._epsilon=0.03      # Default parameter recommended by Cao et al.

        #self._stochasticParam=None


        # Add templates of compiled sympy functions with:
        # 1) Name of the compiled version of the sympy object
        # 2) The function used to generate the underlying sympy object
        #    (convention: starts with "get_", in previous versions have
        #     started with get_ or _compute)



    # # TODO 1: Deterministic solver

    def solve_deterministic(self, t, perf=False, **kwargs):

        result = solve_deterministic(
            x0 = self.initial_state,
            t = t,
            ode_eqns = self.ode,
            trans_rates = self.event_rate_vector,
            n_state = self.num_state,
            n_trans = self.num_events,
            t0=self.initial_time,
            perf = perf,
            **kwargs)

        return result

    def solve_stochastic(
            self,
            t,
            method,
            proc=False,
            perf=False,
            # x0=None,
            # t0=None,
            seed=None,
            iteration=1,
            parallel=False,
            client=None,
            parallel_backend="dask",
            **options):

        # TODO: interface between pygom and solvers or keep as below?...
        # TODO: seeding

        if method == "cao2006":
            options["f_mu"] = self.transition_mean
            options["f_var"] = self.transition_variance

        config = build_config(method, **options)

        # Normalise time input
        if isinstance(t, Number):
            t = float(t)
        elif isinstance(t, (list, tuple)):
            t = np.array(t)
        elif not isinstance(t, np.ndarray):
            raise InputError("Unknown data type for time")

        # Generate quantities prior to simulations:

        # Random seeds
        master_rng = np.random.default_rng(seed)

        rng_params = np.random.default_rng(master_rng.integers(2**32))
        rng_solver = np.random.default_rng(master_rng.integers(2**32))

        # seeds for solver
        seeds = rng_solver.integers(0, 2**32 - 1, size=iteration)

        # rng for params
        # self._parameter_store.rng = rng_params

        for param in self._parameter_store.stochastic_parameters.values():
            if isinstance(param.value, CallableParameter):
                print("initialising rng")
                param.value.rng = rng_params

        if not parallel:
            logging.debug("Performing serial simulation")

            out = []
            for i in range(iteration):
                self._parameter_store.new_realisation()

                result = solve_stochastic(
                    self.initial_state,
                    t,
                    self.event_rate_vector,
                    self.state_change_matrix,
                    self.x_min,
                    self.x_max,
                    self.num_events,
                    config=config,
                    t0=self.initial_time,
                    proceed_if_rates_zero=proc,
                    perf=perf,
                    seed=seeds[i])
                
                out.append(result)

            return out

            # return [
            #     solve_stochastic(
            #         self.initial_state,
            #         t,
            #         self.event_rate_vector,
            #         self.state_change_matrix,
            #         self.x_min,
            #         self.x_max,
            #         self.num_events,
            #         config=config,
            #         t0=self.initial_time,
            #         proceed_if_rates_zero=proc,
            #         perf=perf,
            #         seed=seeds[i])
            #     for i in range(iteration)
            # ]

        # ----------------------------------------------------
        # Parallel execution
        # ----------------------------------------------------

        if parallel_backend == "dask":
            logging.debug(f"Using {parallel_backend} for parallel simulation")

            if client is None:
                from dask.distributed import Client
                client = Client()

            futures = []
            for i in range(iteration):
                self._parameter_store.new_realisation()

                result = solve_stochastic(
                    self.initial_state,
                    t,
                    self.event_rate_vector,
                    self.state_change_matrix,
                    self.x_min,
                    self.x_max,
                    self.num_events,
                    config=config,
                    t0=self.initial_time,
                    proceed_if_rates_zero=proc,
                    perf=perf,
                    seed=seeds[i])
                
                futures.append(result)

            return client.gather(futures)

            # futures = [
            #     client.submit(
            #         solve_stochastic,
            #         self.initial_state,
            #         t,
            #         self.event_rate_vector,
            #         self.state_change_matrix,
            #         self.x_min,
            #         self.x_max,
            #         self.num_events,
            #         config=config,
            #         t0=self.initial_time,
            #         proceed_if_rates_zero=proc,
            #         perf=perf,
            #         seed=seeds[i])
            #     for i in range(iteration)
            # ]
            # return client.gather(futures)

        raise ValueError(f"Unknown backend: {parallel_backend}")

    def solve_determ(self, t, iteration=1, parallel=False, full_output=False):
        '''
        Simulate the ode by generating new realization of the stochastic
        parameters and integrate the system deterministically.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        iteration: int
            number of iterations you wish to simulate
        parallel: bool, optional
            Defaults to True
        full_output: bool, optional
            if we want additional information, Y_all in the return,
            defaults to false

        Returns
        -------
        Y: :class:`numpy.ndarray`
            of shape (len(t), len(state)), mean of all the simulation
        Y_all: :class:`np.ndarray`
            of shape (iteration, len(t), len(state))
        '''

        # if our parameters not stochastic, then we are going to
        # throw a warning because trying to  randomly draw parameters
        # when they are set to be constant is just plain stupid

        if not self._parameter_store.has_stochastic_parameters:
            warnings.warn("System only has deterministic parameters, maybe you"
                          "just want to integrate the model using the "
                          "integrate method?")
        if iteration is None:
            raise InputError("Need to specify the number of iterations")
        if t is None:
            raise InputError("Need to specify the time we wish to observe")

        self._odeSolution = self.integrate(t)

        # try to compute the simulation in parallel
        if parallel:
            try:
                # for i in self._stochasticParam:
                #     if isinstance(i, scipy.stats._distn_infrastructure.rv_frozen):
                #         raise Exception("Cannot perform parallel simulation "
                #                         +"using a serialized object as distribution")
                # # check the type of parameter we have as input

                warnings.warn('Parallel computation not fully tested. Please '
                'check a subset before relying on these answers.'
                )

                import dask.bag
                y = list()
                # Generate a list of parameter values (thetas) to calculate 
                for i in range(iteration):
                    self._parameter_store.new_realisation()
                    y.append(self._parameter_store.values)

                def sim(x):
                    self.parameters = x
                    self._setIntegrateTime(t)
                    return self._integrate(self._odeTime, full_output=False)

                xtmp = dask.bag.from_sequence(y)
                solutionList = xtmp.map(sim).compute()
            except Exception: # as e:
                # logging.debug(e)
                # logging.debug("Serial")
                solutionList = [self.integrate(t) for i in range(iteration)]
        else:
            solutionList = [self.integrate(t) for i in range(iteration)]

        # now make our 3D array
        # the first dimension is the number of iteration
        Y = np.dstack(solutionList).mean(axis=2)

        if full_output:
            return Y, solutionList
        else:
            return Y

    def plot(self, sim_X=None, sim_T=None):
        '''
        Plot the results of a simulation

        Takes the output of a function like `solve_stochast`

        Parameters
        ----------
        sim_X: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        sim_T: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        Notes
        -----
        If either sim_X or sim_T are None the this function will attempt to
        plot the deterministic ODE

        If we have 3 states or more, it will always be arrange such
        that it has 3 columns.  Uses the operation from
        :mod:`odeutils`
        '''
        if (sim_X is None) or (sim_T is None):
            return super(SimulateOde, self).plot()
        ode_utils.plot_stoc(sim_X, sim_T, self)
