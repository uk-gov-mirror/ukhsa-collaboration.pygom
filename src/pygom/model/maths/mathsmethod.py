import logging

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


    def __call__(self, state, time, skip_compilation=False):
        '''
        Dunder function so that when added to the ODE system object it acts like a method
        
        Parameters
        ----------
        state: The values for the system states
        time: The
        '''
        # Check to see if we need to compile
        if self.needs_recompile or self._compiled_fn is None:
            self.compile_function()

        # perform the numerical calculation
        return self._compiled_obj(self._parent_ode._getEvalParam(state, time, None))

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
