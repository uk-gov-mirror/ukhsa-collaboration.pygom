from .mathsmethod import NumericMethod

class Jacobian(NumericMethod):
    method_name = 'jacobian'
    def get_equation(self):
        '''
        Returns the jacobian in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''        
        states = [s for s in self._parent_ode._iterStateList()]
        self._Jacobian = self._parent_ode.ode.get_equation().jacobian(states)

        return self._Jacobian