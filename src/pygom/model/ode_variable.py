"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that contains a variable object for the ode

"""
from sympy.physics.units.quantities import Quantity
from sympy import symbols, Symbol
import numpy as np
import re
import keyword
from ._model_errors import InputError
from numbers import Number

class ODEVariable(object):
    """
    A class that defines the variables in our ODE

    Parameters
    ----------
    ID: str
        identifier of the variable
    symbol: sympy.Symbol
        sympy symbolic representation of the variable. Often taken to
        be sympy.Symbol(ID), but not necessarily.
    value:
        Numeric value of variable
    units: str, optional
        what unit the variable takes. Defaults to None.
    real: bool, optional
        if the variable can only be a real number, defaults to True
    limits: tuple(float, float), optional
        minimum and maximum allowed numerical values of the variable
    """
    def __init__(
            self,
            ID:None|str=None, 
            symbol:None|Symbol|str=None,
            value:None|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(0, np.inf)
        ):

        if (ID is None) and (symbol is None):
            raise InputError(
                f"Must specify at least one of ID or symbol"
            )

        if ID is None:
            ID = str(symbol)

        if not isinstance(ID, str):
            raise TypeError("ID must be a string")
        self.ID = ID
        self.real = real
        self.value = value
        self.units = units
        self.limits = limits

        if symbol is None:
            symbol = ID
        self.symbol = symbol
        
    def __str__(self)->str:
        return self.ID

    def __repr__(self)->str:
        return (
            f"ODEVariable("
            f"{self.ID!r}, "
            f"{self.symbol!r}, "
            f"{self.value!r}, "
            f"{self.units!r}, "
            f"{self.limits!r})"
        )
                                                
    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, Symbol):
            return self.symbol == other
        elif isinstance(other, ODEVariable):
            return (
                self.ID == other.ID and \
                self.symbol == other.symbol
            )
        else:
            # raise NotImplementedError('Wrong input type of %s' % type(other))
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __le__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __gt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __ge__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def _generate_symbol(
            self,
            symbol_name: str,
            real:bool=True
        ) -> list:
        """
        Wrapper of sympy.symbols()

        We cannot let sympy build symbols on its own since we have some additional requirements.

        Generate one or more Sympy symbols from variable name(s)

        Parameters
        ----------
        symbol_name: str
            Name of the symbol
        real: bool
            True if real valued

        Returns
        -------
        sympy.Symbol
        """

        _SYMBOL_RULES = (
            "Symbol names must:\n"
            "  - start with a letter\n"
            "  - then contain only letters, digits or underscores"
            # "  - or be in SymPy range notation (e.g. 'y1:4')"
        )

        # _VALID_SYMBOL = re.compile(
        #     r"^[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?$"
        # )

        _VALID_SYMBOL = re.compile(
            r"^[A-Za-z][A-Za-z0-9_]*$"
        )

        # TODO: should we make 't' a protected symbol for time?
        # NOTE: maybe add t at model initialisation and then any later attempts to add t will be blocked?
        if keyword.iskeyword(symbol_name):
            raise InputError(
                f"'{symbol_name}' is a reserved Python keyword"
            )

        if not _VALID_SYMBOL.fullmatch(symbol_name):
            raise InputError(
                f"Invalid symbol name '{symbol_name}'.\n{_SYMBOL_RULES}"
            )

        return symbols(symbol_name, real=real)

    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol:Symbol|str):
        if isinstance(symbol, str):
            symbol = self._generate_symbol(symbol, self.real)
        elif not isinstance(symbol, Symbol):
            raise InputError(
                'The symbol attribute must be of sympy.Symbol or str type'
            )
        self._symbol = symbol

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        if not isinstance(limits, (tuple, list)):
            raise InputError("Limits must be a tuple or list")

        if len(limits) != 2:
            raise InputError(f"Limits should contain exactly 2 values, received {len(limits)}")

        lower, upper = limits

        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise InputError("Limits must be numeric")

        if lower >= upper:
            raise InputError("Lower limit must be strictly less than upper limit")

        self._limits = (lower, upper)









# def _generate_symbol(
#         self,
#         input_value: str | tuple[str, str]
#     ) -> list:
#     """
#     Generate one or more Sympy symbols from variable name(s)

#     Parameters
#     ----------
#     input_value
#         Either:
#             "x"
#             ("x", "real")
#             ("z", "complex")
#             "x1:5" e.g.

#     Returns
#     -------
#     Symbol | list[Symbol]
#     """

#     _SYMBOL_RULES = (
#         "Symbol names must:\n"
#         "  - start with a letter\n"
#         "  - then contain only letters, digits or underscores\n"
#         "  - or be in SymPy range notation (e.g. 'y1:4')"
#     )

#     _VALID_SYMBOL = re.compile(
#         r"^[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?$"
#     )

#     if isinstance(input_value, str):
#         symbol_name = input_value
#         is_real = True
#     elif isinstance(input_value, tuple):
#         if len(input_value) != 2:
#             raise InputError(
#                 f"Expected 2 values, received {len(input_value)}"
#             )

#         symbol_name, assumption = input_value
#         assumption = str(assumption).lower()

#         if assumption in {"real", "true"}:
#             is_real = True
#         elif assumption in {"complex", "false"}:
#             is_real = False
#         else:
#             raise InputError(
#                 f"Unknown symbol assumption '{assumption}'"
#             )
#     else:
#         raise InputError(
#             f"Unsupported type {type(input_value)}"
#         )

#     # TODO: should we make 't' a protected symbol for time?
#     if keyword.iskeyword(symbol_name):
#         raise InputError(
#             f"'{symbol_name}' is a reserved Python keyword"
#         )

#     if not _VALID_SYMBOL.fullmatch(symbol_name):
#         raise InputError(
#             f"Invalid symbol name '{symbol_name}'.\n{_SYMBOL_RULES}"
#         )

#     result = symbols(symbol_name, real=is_real)

#     if isinstance(result, Symbol):
#         return [result]

#     if isinstance(result, tuple):
#         return list(result)

#     raise InputError(
#         f"Unexpected result returned by sympy.symbols: {type(result)}"
#     )