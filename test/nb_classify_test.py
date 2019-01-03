import unittest
import random
import numpy as np
import src.nb_classify as nb
import src.bow_feat as feat
import src.cv_partition as cv
import src.text as text

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.nb_classify_test
'''

class NBTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.train_size, self.test_size, self.reviews_train, self.reviews_test = cv.prepare_data()
        self.full_vocab = feat.get_vocab(self.reviews_train, cutoff_threshold=9)
        self.vocab_length = len(self.full_vocab)
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_save_to_file(self):
        # a, b, c = np.ones(1000), np.zeros(1000), 0.5
        # nb.save_to_file(a, b, c)
        pass
        
    def test_train_nb_classifier(self):
        # print("\ntraining the Naive Bayes classifier w/ unigram & unsmoothing")
        
        # print("\ngenerating the training matrix ...")
        # # training matrix of data
        # train_matrix = feat.bag_words2vec_unigram(self.full_vocab, self.reviews_train)
        # train_class_vector = np.hstack(
        # (feat.get_class_vec('neg', self.train_size), feat.get_class_vec('pos', self.train_size)))

        # print("\ntraining the Naive Bayes classifier ...")
        # # train the Naive Bayes classifier
        # nb.train_nb_classifier(train_matrix, train_class_vector, "None")
        # print("\nthe training process, DONE. ")
        pass
    
    def test_test_nb_classifier(self):
        # print("\ntesting the Naive Bayes classifier w/ unigram & unsmoothing")
        # # parameters for testing
        # i, neg_correct, pos_correct = 0, 0, 0
        # classification_result = [0]*len(self.reviews_test)  # 0 for misclassification

        # print("\ntesting the Naive Bayes classifier ...")
        # # test the classifier with another review
        # for i in range(len(self.reviews_test)):
        #     test_vec = feat.words2vec_unigram(self.full_vocab, self.reviews_test[i])

        #     test_result = nb.test_nb_classifier(test_vec)
        #     # neg review result=0
        #     if i < self.test_size:
        #         print("Test sample %d \nThis review is %d review, 0 for neg" % (i, test_result))
        #         if test_result == 0:
        #             neg_correct += 1
        #             classification_result[i] = 1
        #     # pos review result=1
        #     else:
        #         print("Test sample %d \nThis review is %d review, 1 for pos" % (i, test_result))
        #         if test_result == 1:
        #             pos_correct += 1
        #             classification_result[i] = 1

        # # print overall accuracy
        # neg_accuracy = (neg_correct/self.test_size)
        # pos_accuracy = (pos_correct/self.test_size)
        # print("\naccuracy for negative reviews", neg_accuracy)
        # print("accuracy for positive reviews", pos_accuracy)
        # print("overall accuracy for this classifier", sum(classification_result)/len(classification_result))
        
        # # save classification results to files
        # nb.save_results("unigram", self.vocab_length, self.train_size, "laplace", neg_accuracy, pos_accuracy)
        # print("\nclassification results written to file")
        pass

    def test_sklearn_nb_classifier(self):
        print("\nusing the built-in Naive Bayes classifier in sklearn to validate ...")
        # training matrix of data
        train_matrix = feat.bag_words2vec_unigram(self.full_vocab, self.reviews_train)
        # training vevctor of labels 
        train_class_vector = np.hstack(
        (feat.get_class_vec('neg', self.train_size), feat.get_class_vec('pos', self.train_size)))

        test_reviews_neg, test_reviews_pos = [], []
        for i in range(len(self.reviews_test)):
            if i < self.test_size:
                test_reviews_neg.append(feat.words2vec_unigram(self.full_vocab, self.reviews_test[i]))
            else:
                test_reviews_pos.append(feat.words2vec_unigram(self.full_vocab, self.reviews_test[i]))

        print(np.array(test_reviews_neg).shape)
        print(np.array(test_reviews_pos).shape)

        nb.sklearn_nb_classifier(train_matrix, train_class_vector, test_reviews_neg, test_reviews_pos)


if __name__ == "__main__":
    unittest.main()
