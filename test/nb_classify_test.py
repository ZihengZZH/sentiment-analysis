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

    def test_n_fold_cons(self):
        n_fold, length_data = 10, 1000
        assert len(nb.n_fold_cons(n_fold, length_data)) == n_fold
        # print(nb.n_fold_cons(n_fold, length_data))

    def test_n_fold_RR(self):
        n_fold, length_data = 10, 1000
        assert len(nb.n_fold_cons(n_fold, length_data)) == n_fold
        # print(nb.n_fold_RR(n_fold, length_data))

    def test_ten_fold_consecutive(self):
        # fold_type = consecutive or RR
        nb.ten_fold_crossvalidation('consecutive', 'unigram')
        pass

if __name__ == "__main__":
    unittest.main()
