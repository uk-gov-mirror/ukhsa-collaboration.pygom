import logging

from .._model_errors import InputError

class MathsMethod:
    _compiled_obj = None # Will store the compiled function in child classes
    method_name = None # Should be overloaded in child classes to the method name that the class attaches 
    needs_recompile = True # This is a compile canary. Should be tripped by the parent ode if the system changes
    outType = None # The type of output (matrix or vector) that the compiled expression produces, if None this will be determined automatically
    def __init__(self, parent_ode)->None:
        '''
        Init function

        Parameters
        ---------
        parent_ode: The system to which this maths method is attached
        backend: The compilation backend as suported by sympy
        '''
        # Save a pointer to the parent
        self._parent_ode = parent_ode

        # Use the parent_ode's compiler class (don't want each MM having their own).
        self._SC = parent_ode._SC


    def __call__(self, state, time):
        '''
        Dunder function so that when added to the ODE system object it acts like a method
        
        Parameters
        ----------
        state: The values for the system states
        time: The timepoint to evaluate for
        '''
        # Check to see if we need to compile
        if self.needs_recompile or self._compiled_obj is None:
            self.compile_function()

        # perform the numerical calculation
        return self._compiled_obj(self._getEvalParam(state, time))

    def get_equation(self):
        '''
        Give a symbolic form of the maths method

        Returns
        -------
        A sympy object representing the symbolic form of this method
        '''
        raise NotImplemented('This is the base class, implement this in a child class!')
    
    def compile_function(self):
        '''
        Compile the symbolic form so that rapid numerical evaluation may occur
        '''
        logging.debug(f'Compiling sympy object {self.method_name}.')
        self._compiled_obj=self._SC.compileExprAndFormat(self._parent_ode._sp,
                                                         self.get_equation(),
                                                         modules='mpmath', 
                                                         outType=self.outType)
        self.needs_recompile = False

    def _getEvalParam(self, state, time):
        if state is None or time is None:
            raise InputError("Have to input both state and time")

        elif not hasattr(self._parent_ode, "_parameters") or self._parent_ode._parameters is None:
            if self._parent_ode.num_param != 0:
                raise InputError("Have not set the parameters yet")

        if hasattr(state, '__iter__'):
            # just in case this isn't a list already
            eval_param = list(state) + [time]
        else:
            eval_param = [state] + [time]

        return eval_param + self._parent_ode._paramValue
