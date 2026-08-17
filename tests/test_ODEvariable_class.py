from pygom import ODEVariable, InputError, _generate_symbol
import sympy
# import pytest

def test_default_symbol_created():
    var = ODEVariable("x")
    assert str(var.symbol) == "x"

# def test_symbol_must_be_sympy_symbol():
#     with pytest.raises(InputError):
#         ODEVariable("x", symbol="x")

def test_string_equality():
    assert ODEVariable("x") == "x"

def test_symbol_equality():
    assert ODEVariable("x") == sympy.Symbol("x")

def test_different_ids_not_equal():
    assert ODEVariable("x") != ODEVariable("y")

def test_repr():
    repr(ODEVariable("x"))