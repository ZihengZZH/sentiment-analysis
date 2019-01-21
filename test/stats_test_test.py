import unittest
import math
import numpy as np
import src.stats_test as st

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.sign_test_test
'''

class StatsTestTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        pass
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass
    
    def test_get_p_value(self):
        assert st.get_combination(6,3) == 20
        assert round(st.get_p_value(3, 11, 10, ignore_ties=True), 3) == .057
        assert round(st.get_p_value(5, 15, 10, ignore_ties=True), 3) == .041
        # print(round(st.get_p_value(3, 11, 10), 3))
        # print(round(st.get_p_value(5, 15, 10), 3))

    def test_run_sign_test(self):
        pass
    
    def test_run_permutation_test(self):
        sys_A_result = np.array(([0.01, 0.03, 0.05, 0.01, 0.04, 0.02]))
        sys_B_result = np.array(([0.1, 0.15, 0.2, 0.08, 0.3, 0.4]))
        p_value, no_larger = st.run_permutation_test(sys_A_result, sys_B_result)
        assert round(p_value, 4) == 0.0462
        assert no_larger == 2
    
    def test_run_monte_carlo_permutation_test(self):
        sys_A_result = np.array(([0.01, 0.03, 0.05, 0.01, 0.04, 0.02]))
        sys_B_result = np.array(([0.1, 0.15, 0.2, 0.08, 0.3, 0.4]))
        p_value, no_larger = st.run_permutation_test(sys_A_result, sys_B_result, R=30)
        print("#larger samples: ", no_larger)
        print("p-value: ", round(p_value, 4))

if __name__ == "__main__":
    unittest.main()

'''

The calculator in GraphPad used #'success' and #trails to calcualte p-value
It does not take the null/ties into account (JUST IGNORE THE TIES)

However, disregarding ties will tend to affect a study's statistical power
Here we treat ties by adding 0.5 events to the positive and 0.5 events to the negative slide (and round up at the end). Therefore, the p-value could be a little different that calculated from the QuickCalcs or read from the Table D.

'''