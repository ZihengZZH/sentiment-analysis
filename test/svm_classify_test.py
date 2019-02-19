import unittest
import random
import numpy as np
import src.svm_classify as svm
import src.bow_feat as feat
import src.cv_partition as cv
import src.text as text
from collections import namedtuple

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
        svm.prepare_data(train_matrix, self.train_size, False)
        svm.prepare_data(test_matrix, self.test_size, False, test=True)
        print("\ndata preparation, DONE")
        pass

    def test_json_load(self):
        parameters = namedtuple("Parameters", "kernel C gamma")
        svm_para_dbow = parameters(2, 100, 0.001)
        svm_para_dm = parameters(2, 10, 0.0001)
        svm_para_unigram = parameters(0, 1, 0)
        svm_para_bigram = parameters(2, 10, 0.001)
        self.assertEqual(svm.svm_para_dbow, svm_para_dbow)
        self.assertEqual(svm.svm_para_dm, svm_para_dm)
        self.assertEqual(svm.svm_para_unigram, svm_para_unigram)
        self.assertEqual(svm.svm_para_bigram, svm_para_bigram)

if __name__ == "__main__":
    unittest.main()