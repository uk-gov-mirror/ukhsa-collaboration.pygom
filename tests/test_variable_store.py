from unittest import main, TestCase

import time
import sympy

from pygom.model.ode_utils import VariableStore


class TestVariableStore(TestCase):

    def setUp(self):
        pass

    def test_adding_variable(self):
        # create a store
        vs = VariableStore()
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')

        # add a variables one as a string, a named symbol, a raw symbol, as dict
        vs.append('a')
        vs.append('b', symbol=b)
        vs.append(c)
        vs['d'] = d


        # The zeroth variable symbol should be 'a', first b, third c, forth d
        self.assertEqual(str(vs.index[0].symbol), 'a')
        self.assertEqual(vs.index[1].symbol, b)
        self.assertEqual(vs.index[2].symbol, c)
        self.assertEqual(vs.index[3].symbol, d)

        # c should be called c
        self.assertEqual(str(vs.index[2].symbol), 'c')

        # But none should have a value
        self.assertIsNone(vs.index[0].value)
        self.assertIsNone(vs.index[1].value)
        self.assertIsNone(vs.index[2].value)
        self.assertIsNone(vs.index[3].value)

        # We should be able to call by name as well as position
        self.assertEqual(vs.index[0].symbol, vs['a'].symbol)
        self.assertEqual(vs.index[1].symbol, vs['b'].symbol)
        self.assertEqual(vs.index[2].symbol, vs['c'].symbol)
        self.assertEqual(vs.index[3].symbol, vs['d'].symbol)

        # Test updating a variable
        vs['b'] = c
        self.assertEqual(vs.index[2].symbol, vs['b'].symbol)
        self.assertEqual(vs.getIndex('b'), 1)

    def test_adding_list(self):
        # create the store
        vs = VariableStore()
        # create some symbols
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')

        # Create a mixed list
        variables = ['a', b, c, d]

        # add the list to the store
        vs.extend(variables)

        # The zeroth variable symbol should be 'a', first b, third c, forth d
        self.assertEqual(str(vs.index[0].symbol), 'a')
        self.assertEqual(vs.index[1].symbol, b)
        self.assertEqual(vs.index[2].symbol, c)
        self.assertEqual(vs.index[3].symbol, d)

        # c should be called c
        self.assertEqual(str(vs.index[2].symbol), 'c')

        # But none should have a value
        self.assertIsNone(vs.index[0].value)
        self.assertIsNone(vs.index[1].value)
        self.assertIsNone(vs.index[2].value)
        self.assertIsNone(vs.index[3].value)


    def test_assigning_values_singles(self):
        # create the store
        vs = VariableStore()

        # Create a list of variables
        variables = ['a', 'b', 'c', 'd']

        # Create a list of values
        values = [2, 3, 5, 7]

        # add the variables to the store
        vs.extend(variables)

        # assign the variables (in reverse order)
        for variable, value in reversed(list(zip(variables, values))):
            vs.set_value(variable=variable, value=value)

        # Test that the values match up
        for variable, value in zip(variables, values):
            self.assertEqual(vs[variable].value, value)

    def test_assigning_values_list(self):
        # create the store
        vs = VariableStore()

        # Create a list of variables
        variables = ['a', 'b', 'c', 'd']

        # Create a list of values
        values = [2, 3, 5, 7]

        # add the variables to the store
        vs.extend(variables)

        # assign the variables (in reverse order)
        vs.set_value_list(values)

        # Test that the values match up
        for variable, value in zip(variables, values):
            self.assertEqual(vs[variable].value, value)

    def test_solution_speed(self):
        # create the store
        vs = VariableStore()

        # how big a system to test on
        N = int(1E5) # one hundred thousand parameters / states!
        ALLOWABLE_VAR_CREATION_TIME = 10 # give it 10 seconds to create vars
        ALLOWABLE_VALUE_UPDATE = 1 # 1 second to update the values of vars

        start_time = time.perf_counter()
        # add the variables
        vs.extend(['var_' + n for n in map(str, range(N))])
        value_time = time.perf_counter()
        vs.set_value_list(list(range(N)))
        end_time = time.perf_counter()

        self.assertLess(value_time - start_time, ALLOWABLE_VAR_CREATION_TIME)
        self.assertLess(end_time - value_time, ALLOWABLE_VALUE_UPDATE)


# TODO: Test error conditions
# repeat adding of variable
# mismatched values list length
# non-str keys

    def tearDown(self):
        pass


if __name__ == '__main__':
    main()
