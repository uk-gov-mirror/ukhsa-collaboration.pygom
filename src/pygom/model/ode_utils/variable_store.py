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
#from collections.abc import MutableMapping

#from collections import OrderedDict
from indexed import IndexedOrderedDict

from sympy import Symbol

from ..ode_variable import ODEVariable

__all__ = ['VariableStore']

class IndexShim(object):
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, item:int):
        return self.parent._variables.values()[item]
        

class VariableStore(object):
    def __init__(self, storage_type:str='variable'):
        '''
        The init method
        '''
        self._variables = IndexedOrderedDict()
        self._variable_pos = dict()
        self.storage_type = storage_type
        self.index = IndexShim(parent=self)
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
        Add a variable to the store

        Parameters
        ----------
        variables: A list of variables to add, either as a list of strings . These will be appended at the end
          of the list of variables

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



    def set_value(self, variable:str, value:float) -> None:
        '''
        Set the value of a variable

        Parameters
        ----------
        variable: The name of the variable as a string
        value: The value that the variable should take (numeric).

        '''
        self[variable].value = value
    
    def set_value_list(self, variables:list) -> None:
        '''
        Set the value of all the variables

        Parameters
        ----------
        Variables: A list, the same length as the number of variables, 
          containing the values 
        '''
        if len(variables) != len(self):
            raise ValueError(F'The length of the supplied list of values must '
                             f'match the number of {self.storage_type}. '
                             f'Expected {len(self)}, got {len(variables)}.')
        
        for key, variable in zip(self._variables.keys(), variables):
            self[key].value = variable

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
