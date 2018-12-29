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
        # _ = nb.nb_classifier('unigram', 'laplace')
        # _ = nb.nb_classifier('bigram', 'laplace')
        # _ = nb.nb_classifier('both', 'laplace', test=True)
        
        # _ = nb.nb_classifier('unigram', 'None')
        # _ = nb.nb_classifier('bigram', 'None')
        # _ = nb.nb_classifier('both', 'None')
        pass


if __name__ == "__main__":
    unittest.main()
