'''
An object to hold variables for a PyGOM ODE system

This object is designed to:
* Store all the variables
* Provide a list of variables
  * as symbols
  * as values
* Rapidly set a variable value from a list
* Provide the index of a named parameter

'''
from types import NoneType
from indexed import IndexedOrderedDict

from sympy import Symbol
import numpy as np

from ..ode_variable import ODEVariable
from .._model_errors import InputError

from scipy.stats._distn_infrastructure import rv_frozen

__all__ = ['VariableStore','ParameterStore']

class IndexShim(object):
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, item:int):
        return self.parent._variables.values()[item]
class VariableStore(object):
    def __init__(self, 
                 storage_type:str='variable', 
                 acceptable_value_types:list=[float, int]):
        '''
        The init method
        '''
        self._variables = IndexedOrderedDict()
        self._variable_pos = dict()
        self.storage_type = storage_type
        self.index = IndexShim(parent=self)

        # Type checking for the values assigned to a parameter
        acceptable_value_types.append(NoneType) # None is always ok
        self.acceptable_value_types = {avt: avt.__name__ for avt in 
                                       acceptable_value_types}
        self._values_by_type = {key: dict() for key in 
                                self.acceptable_value_types.values()}

        #self.sibling_lists = []

    def __getitem__(self, item:str):
        '''
        Getter when referencing the variable by name
        '''
        return self._variables[item]

    def __setitem__(self, key:str, value:str|Symbol|ODEVariable) -> None:
        '''
        Setter when referencing by name
        '''
        # Self defence, IDs have to be a string
        if not isinstance(key, str):
            raise TypeError(f'{self.storage_type} IDs must be of str type, was'
                            f'{type(key)}.')
        
        # convert the value to an ODEVariable
        var_obj:ODEVariable = self._check_variable(variable=value)

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

    def __len__(self)->int:
        '''
        The current length of the store
        '''
        return len(self._variables)
    
    def __str__(self)->str:
        '''
        gives a string representation of the parameters
        '''
        return ', '.join(self._variables.keys())
    
    def __contains__(self, key)->bool:
        if not isinstance(key, str):
            raise TypeError(f'{self.storage_type} IDs must be of str type, was'
                            f'{type(key)}.')
        return key in self._variables

    def get_index(self, key:str)->int:
        '''
        Get the index of a particular variable 
        
        This should be fast - O(1)
        '''
        return (self._variable_pos[key])

    def append(self, 
               variable:str|Symbol|ODEVariable, 
               symbol:Symbol|None=None, 
               real:bool=True) -> ODEVariable:
        '''
        Add a variable to the store

        Parameters
        ----------
        variable: The name of variable to add. This will be appended at the end
          of the list of variables

        '''
        var_obj:ODEVariable = self._check_variable(variable=variable,
                                                   symbol=symbol,
                                                   real=real
                                                   )

        #TODO: Check that this new variable is not in the sibling lists

        if var_obj.ID in self._variables:
            raise AttributeError(f'You may not add a {self.storage_type} more '
                                 f'than once. {var_obj.ID} already exists.'
                                 )
        # Store the new variable
        self._variables[var_obj.ID] = var_obj
        # And the position into which we put it
        self._variable_pos[var_obj.ID] = len(self._variables) - 1
    
    def extend(self, variables:list) -> None:
        '''
        Add a list of variables to the store

        Parameters
        ----------
        variables: A list of variables to add, either as a list of strings or
        ODEVariables. These will be appended at the end of the list of variables

        '''
        for variable in variables:
            # Quick / safe way to generate the ID although we will "check" twice
            var_obj:ODEVariable = self._check_variable(variable=variable)
            self[var_obj.ID] = var_obj
    
    def _check_variable(self, 
                        variable:str|Symbol|ODEVariable,
                        symbol:Symbol|None=None, 
                        real:bool=True
                        )->ODEVariable:
            # Turn variable into a ODEVariable if required
        # TODO: Surface the units part of ODEVariable
        if isinstance(variable, str):
            var_obj = ODEVariable(ID=variable, 
                                  symbol=symbol,
                                  real=real
                                  )
        elif isinstance(variable, Symbol):
            var_obj = ODEVariable(ID=str(variable),
                                  symbol=variable,
                                  real=real
                                  )
        elif isinstance(variable, ODEVariable):
            var_obj = variable
        else:
            raise InputError(f'You may not add an object of type '
                             f'{type(variable)} as a parameter.')
        
        return var_obj



    def set_value(self, variable:str, value) -> None:
        '''
        Set the value of a variable

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The value that the variable should take.

        '''
        # Book-keeping for the by-type dicts
        current_type = ''
        for at, atn in self.acceptable_value_types.items():
            if isinstance(self[variable].value, at):
                current_type = atn

        new_type = ''
        for at, atn in self.acceptable_value_types.items():
            if isinstance(value, at):
                new_type = atn
        

        if new_type != current_type:
            self._values_by_type[current_type].pop(variable, None)
           
            if new_type == '':
                raise InputError(f'You may not add an object of type {type(value).__name__}'
                                 f' as a value for a {self.storage_type}.'
                                 f' Only {list(self.acceptable_value_types.keys())} are '
                                 'permitted (or sub-classes).')
            
            # Set a pointer to the new location
            self._values_by_type[new_type] [variable]=self._variables[variable]

        # Set the value   
        self[variable].value = value
    
    def set_value_list(self, values:list) -> None:
        '''
        Set the value of all the variables

        Parameters
        ----------
        Values: A list, the same length as the number of variables, 
          containing the values 
        '''
        if len(values) != len(self):
            raise ValueError(F'The length of the supplied list of values must '
                             f'match the number of {self.storage_type}. '
                             f'Expected {len(self)}, got {len(values)}.')
        
        for key, value in zip(self._variables.keys(), values):
            self.set_value(key, value)
        
    def set_value_dict(self, values:dict):
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
    def all_values_set(self)->bool:
        '''
        Have all the values been set?
        '''
        return len(self._values_by_type[NoneType.__name__]) == 0
    
    @property
    def variables(self)->list[str]:
        return [variable.ID for variable in self._variables.values()]

    @property
    def values(self)->list[float]:
        '''
        Get a list of all the values stored
        '''
        return [variable.value for variable in self._variables.values()]
    
    @values.setter
    def values(self,
               values:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        """
        Set the values for the parameters already defined.  Note that unless
        the parameters are entered via a dictionary or a two element list,tuple
        we assume that it is in the order of :meth:`.getParamList`

        Parameters
        ----------
        parameters: dict of {parameter_ID: parameter_value} (prefered) _or_
            a list which contains elements made of 2 element tuples 
            (string, numeric value) _or_ a single array like object with
            length equal to the number of parameters, in the same order as they
            were created.
        """
        # Either a list or a dict
        if isinstance(values, (list, tuple, np.ndarray)):
            # Looks like a list but is it a dict in disguise (list of tuples)?
            if len(values) > 0:
                if isinstance(values[0], tuple) and len(values[0]) == 2:
                    # do we have at least one tuple of length 2?
                    try:
                        values = {key: value for key, value in values}
                    except ValueError as e:
                        raise ValueError(f'The {self.storage_type} list' 
                                         ' supplied looked like a list of'
                                         ' tuples, (NAME, VALUE) and'
                                         ' PyGOM tried to evaluate it on that'
                                         ' basis but these entries '
                                        f'{[value for value in values 
                                            if len(value)!=2]}',
                                        ' were not of length 2,'
                                        ' please check these.') from e
                    # Set as dict
                    self.set_value_dict(values)
                else:
                    # Not a dict in disguise, set as a list
                    self.set_value_list(values)
        elif isinstance(values, dict):
            # This is the way, a eplicit dict of [ID: value]
            self.set_value_dict(values)
        else:
            raise InputError(f'Expecting a dict, or iterable '
                             f'input not {type(values)}')

    def symbol_list(self)->list[Symbol]:
        '''
        Get a list of all the symbols stored
        '''
        return [variable.symbol for variable in self._variables.values()]
    
    def symbol_dict(self)->dict[str: Symbol]:
        '''
        Get a dict of all the symbols stored
        '''
        return {variable.ID: variable.symbol for variable in self._variables.values()}

class ParameterStore(VariableStore):
    def __init__(self)->None:
        super().__init__(storage_type='parameter',
                         acceptable_value_types=[int,
                                                 float, 
                                                 rv_frozen
                                                 ]
                         )