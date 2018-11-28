import unittest
import random
import src.nb_classify as nb

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.nb_classify_test
'''

class NBTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        pass
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_nb_classifier(self):
        nb.nb_classifier('laplace', True, True) # laplace + freq
        nb.nb_classifier('laplace', False, True) # laplace + pres
        nb.nb_classifier('None', True, True) # no smoothing + freq
        nb.nb_classifier('None', False, True) # no smoothing + pres

if __name__ == "__main__":
    unittest.main()
