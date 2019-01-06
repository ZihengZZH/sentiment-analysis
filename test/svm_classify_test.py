import unittest
import random
import numpy as np
import src.svm_classify as svm
import src.bow_feat as feat
import src.cv_partition as cv
import src.text as text

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.svm_classify_test
'''

class SVMTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.train_size, self.test_size, self.reviews_train, self.reviews_test = cv.prepare_data()
        self.full_vocab = feat.get_vocab(self.reviews_train, cutoff_threshold=9)
        self.vocab_length = len(self.full_vocab)

    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_prepare_data_svm(self):
        print("\ngenerating the training matrix ...")
        # training matrix of data
        train_matrix = feat.bag_words2vec_unigram(self.full_vocab, self.reviews_train)
        print("\ngenerating the testing matrix ...")
        # testing matrix of data
        test_matrix = feat.bag_words2vec_unigram(self.full_vocab, self.reviews_test)
        
        print("\nprepare the data for the SVM-Light classifier ...")
        svm.prepare_data_svm(train_matrix, self.train_size)
        svm.prepare_data_svm(test_matrix, self.test_size, test=True)
        print("\ndata preparation, DONE")
        pass

    def test_svm_classifier(self):
        svm.train_svm_classifier()
        svm.test_svm_classifier()
        pass
        

if __name__ == "__main__":
    unittest.main()