from .mathsmethod import MathsMethod

class Jacobian(MathsMethod):
    method_name = 'jacobian'
    def get_equation(self):
        '''
        Returns the jacobian in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''        
        self._parent_ode.get_ode_eqn()
        states = [s for s in self._parent_ode._iterStateList()]
        self._Jacobian = self._parent_ode._ode.jacobian(states)

        return self._Jacobian