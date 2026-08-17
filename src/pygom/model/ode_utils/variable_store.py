'''

Variable (state and parameter) registry

An object to hold variables for a PyGOM compartmental model system

This object is designed to:
* Store all the variables
* Provide a list of variables
  * as symbols
  * as values
* Rapidly set a variable value from a list
* Provide the index of a named parameter

needs to provide:
- namespace
- values


'''

from types import NoneType
from collections import OrderedDict
from indexed import IndexedOrderedDict

import numpy as np

from sympy import Symbol, symbols

from pygom.model.ode_variable import ODEVariable
from pygom.model._model_errors import InputError

from scipy.stats._distn_infrastructure import rv_frozen

__all__ = ['VariableStore','ParameterStore', 'StateStore']

class IndexShim(object):
    """
    TODO: docstring
    """
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, item:int):
        return self.parent._variables.values()[item]

class VariableStore(object):

    '''

    Variable (state and parameter) registry

    An object to hold variables for a PyGOM compartmental model system

    This object is designed to:
    * Store all the variables
    * Provide a list of variables
        * as symbols
        * as values
    * Rapidly set a variable value from a list
    * Provide the index of a named parameter
    * Manage variables e.g. duplicates

    needs to provide:
    - namespace
    - values

    Parent class for parameter and state stores

    TODO:
    1) docsting for this
    2) different default limits for states and variables
    '''

    def __init__(
            self,
            storage_type:str='variable', 
            acceptable_value_types:list=[float, int]
        ):
        '''
        The init method
        '''
        self._variables = IndexedOrderedDict()
        self._variable_pos = dict()
        self.storage_type = storage_type
        self.index = IndexShim(parent=self)

        # Type checking for the values assigned to a parameter
        acceptable_value_types.append(NoneType) # None is always ok
        self.acceptable_value_types = {
            avt: avt.__name__ for avt in acceptable_value_types
        }
        self._values_by_type = {
            key: dict() for key in self.acceptable_value_types.values()
        }

    def __getitem__(self, item:str) -> ODEVariable:
        '''
        Getter when referencing the variable by name
        '''
        return self._variables[item]

    def __len__(self)->int:
        '''
        The current number of variables in the store
        '''
        return len(self._variables)
    
    def __contains__(self, key) -> bool:
        # if not isinstance(key, str):
        #     raise TypeError(
        #         f'{self.storage_type} IDs must be of str type, was {type(key)}.')
        return key in self._variables

    def __setitem__(self, key:str, var_obj:ODEVariable) -> None:
        # TODO: have assumed that this is called from within class after var_obj has been
        #       verified as ODEvariable.
        '''
        Setter when referencing by name
        '''

        # Self defence, IDs have to be a string
        if not isinstance(key, str):
            raise TypeError(
                f'{self.storage_type} IDs must be of str type, was {type(key)}.'
            )
        
        # # convert the value to an ODEVariable
        # var_list:list[ODEVariable] = self._check_variable(variable=value)

        #TODO: Check that this new variable is not in the sibling lists

        # check to see if we need to record the position of this key
        # and record it if we do
        if key not in self._variables:
            self._variable_pos[key] = len(self._variables) 

        # Store the new / updated variable
        self._variables[key] = var_obj

        # re-record the value
        value = var_obj.value

        # deal with the bootstrapping problem (everything goes in None to start)
        self._values_by_type[NoneType.__name__] [key]=self._variables[key]

        # Properly log the real value (maintains book-keeping)
        self.set_value(key, value)

    def get_index(self, key:str) -> int:
        '''
        Get the index of a particular variable 
        
        This should be fast - O(1)
        '''
        return self._variable_pos[key]

    def _check_variable(
            self,
            variable:str|Symbol|ODEVariable
        ) -> ODEVariable:
        '''
        TODO: rename this method to _build_variable?

        Normalise any user-provided representation of a variable (or variables) into a list of ODEVariable objects

        Parameters
        ----------
        variable: str | sympy.Symbol | ODEVariable
            String, symbolic or ODEVariable representation of variable
        symbol: sympy.Symbol (optional)
            Symbolic representation of variable
        real: bool
            True if variable is real
        limits: tuple[number, number]
            Minimum and maximum allowed values.
        '''

        if isinstance(variable, ODEVariable):
            return variable
        # TODO: why allow users to specify string, sympy symbols or ODEvars?
        #       seems like too many options that makes this bit awkward 
        #       Do we want users bringing sympy objects into pygom themselves?
        elif isinstance(variable, str):
            return ODEVariable(ID=variable)
        elif isinstance(variable, Symbol):
            return ODEVariable(symbol=variable)
        else:
            raise InputError(
                f'You may not add an object of type '
                f'{type(variable)} as a {self.storage_type}.'
            )

    def add(
            self, 
            variable:str|Symbol|ODEVariable|list
        ):
        '''
        Add a variable(s) to the store

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
            of the list of variables
        '''

        if isinstance(variable, (str, Symbol, ODEVariable)):
            variable = [variable]

        var_list = [self._check_variable(var) for var in variable]

        for var_obj in var_list:
            if var_obj.ID in self._variables:
                raise InputError(
                    f'You may not add a {self.storage_type} more '
                    f'than once. {var_obj.ID} already exists.'
                )
            self[var_obj.ID] = var_obj

    def _get_value_type(self, value):
        for at, atn in self.acceptable_value_types.items():
            if isinstance(value, at):
                return atn
        return None

    def set_value(self, variable:str, value) -> None:
        '''
        Set the value of a variable

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The value that the variable should take.
        '''
        current_value = self[variable].value

        current_type = self._get_value_type(current_value)
        new_type = self._get_value_type(value)

        if new_type is None:
            raise InputError(
                f'You may not add an object of type {type(value).__name__}'
                f' as a value for a {self.storage_type}.'
                f' Only {list(self.acceptable_value_types.keys())} are '
                'permitted (or sub-classes).')

        if current_type != new_type:
            self._values_by_type[current_type].pop(variable, None)
            # Set a pointer to the new location
            self._values_by_type[new_type][variable] = self._variables[variable]

        # Set the value   
        self[variable].value = value

    @property
    def all_values_set(self)->bool:
        '''
        Have all the values been set?
        '''
        return len(self._values_by_type[NoneType.__name__]) == 0
    
    @property
    def variables(self)->list[str]:
        '''
        Get a list of strings of the names for all the variables 
        '''
        return [variable.ID for variable in self._variables.values()]

    @property
    def values(self)->list[float]:
        '''
        Get a list of all the values stored
        '''
        return [variable.value for variable in self._variables.values()]
    
    @property
    def values_full(self)->list[float]:
        '''
        Get a list of all the values stored as ODEVariable objects
        '''
        return [variable for variable in self._variables.values()]

    # TODO: Try just supporting dict input
    @values.setter
    def values(
        self,
        values:dict[str: float]
    ) -> None:
        '''
        Set the value of the variables

        This is explicit and so the prefered way to set the variable values.

        Parameters
        ----------
        Values: A dict keyed on the variable name with value equal to the value.
        ''' 
        for key, value in values.items():
            self.set_value(key, value)

    @property
    def symbol_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored in the order they were added
        '''
        return [variable.symbol for variable in self._variables.values()]
    
    @property
    def symbol_dict(self)->dict[str: Symbol]:
        '''
        Get a OrderedDict of all the symbols stored, keyed on the str 
        representation and value equal to the symbol
        '''
        result = OrderedDict()

        for variable in self._variables.values():
            result[variable.ID] = variable.symbol
        return result

class CallableParameter:
    def __init__(
            self,
            value:tuple[callable, dict|tuple],
            rng:np.random._generator.Generator=None
        ):
        """
        Parameters
        ----------
        value: tuple[callable, dict|tuple]
            value[0] is the probability distribution and value[1] the function parameters
        rng: np.random._generator.Generator
            Numpy random number generator
        """

        if not callable(value[0]):
            raise InputError("First element must be callable.")
        self._callable = value[0]

        # parse args/kwargs
        if isinstance(value[1], dict):
            self.args = []
            self.kwargs = value[1]
        elif isinstance(value[1], tuple):
            self.args = value[1]
            self.kwargs = {}
        else:
            raise InputError(
                'Second element should be either a tuple or a '
                'dict when using multi-argument distribution '
                f'definition. Type of input was {type(value[1])}.'
            )
        self.rng = rng

    def __call__(self, n=1):
        """
        Call the underlying function.

        If the function accepts an `rng` argument, pass it.
        Otherwise fall back to the old behavior.
        """

        return self._callable(n, *self.args, rng=self.rng, **self.kwargs)
        # try:
        #     # Try passing rng explicitly
        #     return self._callable(n, *self.args, rng=self.rng, **self.kwargs)
        # except TypeError:
        #     # Function did not accept rng -> backwards compatible path
        #     return self._callable(n, *self.args, **self.kwargs)

class ParameterStore(VariableStore):
    '''
    A class to store parameters of an ODE system

    This is a specialised version of VariableStore which is able to handle
    values of a parameter that are draws from a stochatic distribution.
    '''
    def __init__(self)->None:
        super().__init__(
            storage_type='parameter',
            acceptable_value_types=[
                int,
                float,
                rv_frozen,
                CallableParameter
            ]
        )
        self._realisation_vals = None

    def set_value(self, variable, value):
        '''
        Sets the value of a variable
        '''
        # convert callables nested in tuples into callables class
        if isinstance(value, tuple):
            value = CallableParameter(value)

        return super().set_value(variable, value)
        
    @property
    def has_stochastic_parameters(self)->bool:
        '''
        Simple check to see if there are any stochastic parameters in the store
        '''
        return (
            len(self._values_by_type.get(rv_frozen.__name__, {}))  + 
            len(self._values_by_type.get(CallableParameter.__name__, {}))
        ) != 0
        
    @property
    def stochastic_parameters(self)->dict[str: rv_frozen]:
        '''
        Provides a dict of stochastic parameters (i.e. ones where the variable)
        has been defined as an instance of rv_frozen.

        Returns
        -------
        Dict keyed on parameter name with value = the distribution
        '''
        result = self._values_by_type[rv_frozen.__name__].copy()
        result.update(self._values_by_type[CallableParameter.__name__])
        return result
    
    def new_realisation(self)->None:
        '''
        Generate a new realiasation of the parameters
        '''
        # Just wipe the cache
        self._realisation_vals = None

    @property
    def values(self)->list[float]:
        '''
        Provides the values for the parameters
        
        If there are stochastic parameters then a draw will be made and stored
        and returned on subsequent calls to this method.

        To generate a new realisation call new_realisation.

        Returns
        -------
        A list of numeric values.

        For each element in the list if a parameter isstochastic then a new 
        value is drawn, if it is deterministic then the value is simply added. 
        '''
        # Check the cache for an existing set (and return that)
        if self._realisation_vals is not None:
            return self._realisation_vals
        
        # Build a new parameter set
        result = list()

        # handle the different ways in which a stochastic parameter can get a 
        # new realisation
        for parameter in self._variables.values():
            if isinstance(parameter.value, rv_frozen):
                result.append(parameter.value.rvs(1, random_state=self.rng)[0])
            elif isinstance(parameter.value, CallableParameter):
                result.append(parameter.value())
            else:
                # The deterministic case
                result.append(parameter.value)
        
        #cache the result
        self._realisation_vals = result
        
        return result
    
    @values.setter
    def values(
        self,
        values:dict[str: float]|list[tuple[str,float]]|list[float]
    ) -> None:
        # set the values via the parent property
        VariableStore.values.fset(self, values)

        # Reset the cache
        self._realisation_vals = None

class StateStore(VariableStore):
    '''
    A class to store states of an ODE system
    '''
    def __init__(self)->None:
        super().__init__(
            storage_type='state',
            acceptable_value_types=[
                int,
                float
            ]
        )
        self._realisation_vals = None