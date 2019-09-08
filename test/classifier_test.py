import unittest
import random
import numpy as np
from sklearn import datasets
from src.classifier.nb_classify import NBClassifier
from src.classifier.svm_classify import SVMClassifier
from src.classifier.rf_classify import RFClassifier
from src.embeddings.bow import BagOfWords
from src.utils.data import load_data


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        data_iris = datasets.load_iris()
        X = data_iris.data
        y = data_iris.target
        indices = np.random.permutation(len(X))
        test_size = 15
        self.X_train = X[indices[:-test_size]]
        self.y_train = y[indices[:-test_size]]
        self.X_test = X[indices[-test_size:]]
        self.y_test = y[indices[-test_size:]]

    def test_svm_classifier(self):
        svm = SVMClassifier('test')
        svm.load_data(self.X_train, self.y_train, self.X_test, self.y_test)
        svm.train_model()
        svm.evaluate_model()
    
    def test_rf_classifier(self):
        rf = RFClassifier('test')
        rf.load_data(self.X_train, self.y_train, self.X_test, self.y_test)
        rf.train_model()
        rf.evaluate_model()


if __name__ == "__main__":
    unittest.main()
