import unittest
import src.cv_partition as cv

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.cv_partition_test
'''

class CVPartitionTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.toy_neg_reviews = [x for x in range(1000)]
        self.toy_pos_reviews = [x for x in range(1000)]
        self.rr_neg_reviews = [(x,x) for x in range(1000)]
        self.rr_pos_reviews = [(x,x) for x in range(1000)]
        pass
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_partition(self):
        train_size, test_size, reviews_train, reviews_test = cv.partition(self.toy_neg_reviews, self.toy_pos_reviews)
        assert train_size + test_size == 1000
        assert len(reviews_train) == train_size*2
        assert len(reviews_test) == test_size*2
        assert set(reviews_train[:train_size] + reviews_test[:test_size]) == set(range(1000))
        assert set(reviews_train[train_size:] + reviews_test[test_size:]) == set(range(1000))

    def test_prepare_data_tenfold(self):
        test_range_tenfold = cv.n_fold_cons(10, 1000)
        for test_range in test_range_tenfold:
            train_size, test_size, reviews_train, reviews_test = cv.prepare_data_tenfold(self.toy_neg_reviews, self.toy_pos_reviews, test_range)
            assert train_size + test_size == 1000
            assert len(reviews_train) == train_size*2
            assert len(reviews_test) == test_size*2
            assert set(reviews_train[:train_size] + reviews_test[:test_size]) == set(range(1000))
            assert set(reviews_train[train_size:] + reviews_test[test_size:]) == set(range(1000))
    
    def test_prepare_data_roundrobin(self):
        test_splits_roundrobin = cv.n_fold_RR(10, 1000)
        for test_range in test_splits_roundrobin:
            train_size, test_size, reviews_train, reviews_test = cv.prepare_data_roundrobin(self.rr_neg_reviews, self.rr_pos_reviews, test_range)
            assert train_size + test_size == 1000
            assert len(reviews_train) == train_size*4
            assert len(reviews_test) == test_size*4
            assert set(reviews_train[:train_size*2] + reviews_test[:test_size*2]) == set(range(1000))
            assert set(reviews_train[train_size*2:] + reviews_test[test_size*2:]) == set(range(1000))

    def test_prepare_data_gridsearch(self):
        for _ in range(10):
            train_size, test_size, reviews_train, reviews_test = cv.prepare_data(tags=False, gridsearch=True)
            assert train_size + test_size == 1000
            assert len(reviews_train) == train_size*2
            assert len(reviews_test) == test_size*2

if __name__ == "__main__":
    unittest.main()