import unittest
import src.nb_classify as nb
import src.sign_test as st
import math

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.sign_test_test
'''

class SignTestTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        pass
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass
    
    def test_get_p_value(self):
        assert st.get_combination(6,3) == 20
        assert round(st.get_p_value(3, 11, 0), 4) == .0574
        assert round(st.get_p_value(5, 15, 0), 4) == .0414

    def test_run_sign_test(self):
        unigram_result_no_smoothing = nb.nb_classifier('bigram', 'None')
        unigram_result_smoothing = nb.nb_classifier('bigram', 'laplace')
        st.run_sign_test(unigram_result_no_smoothing, unigram_result_smoothing, 'unigram')
        

if __name__ == "__main__":
    unittest.main()

'''
A LITTLE CONFUSED
The calculator in GraphPad used #'success' and #trails to calcualte p-value
It does not take the null/ties into account
However, disregarding ties will tend to affect a study's statistical power
Here we treat ties by adding 0.5 events to the positive and 0.5 events to the negative slide 
(and round up at the end)

'''