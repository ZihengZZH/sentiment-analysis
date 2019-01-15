import unittest
import random
import src.doc2vec as doc2vec
import src.cv_partition as cv

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.doc2vec_test
'''

class doc2vecTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        pass

    # method called immediately after the test method has been called and the result recoreded
    def tearDown(self):
        pass
    
    def test_load_model(self):
        model_4 = doc2vec.load_model(4)
        assert str(model_4) == "Doc2Vec(dbow,d150,n5,mc2,t12)"
        model_7 = doc2vec.load_model(7)
        assert str(model_7) == "Doc2Vec(\"alpha=0.05\",dm-m,d150,n5,w10,mc2,t12)".replace('-', '/')

    def test_infer_embedding(self):
        train_size, test_size, reviews_train, reviews_test = cv.prepare_data()
        model_3 = doc2vec.load_model(3) # vector size 150
        train_vectors, train_labels = doc2vec.infer_embedding(model_3, reviews_train, train_size)
        assert sum(train_labels) == train_size
        assert len(train_vectors[random.randrange(0, train_size)]) == 150
        model_5 = doc2vec.load_model(5) # vector size 100
        test_vectors, test_labels = doc2vec.infer_embedding(model_5, reviews_test, test_size)
        assert sum(test_labels) == test_size
        assert len(test_vectors[random.randrange(0, test_size)]) == 100
        print("training vectors shape: %d * %d" % (len(train_vectors), len(train_vectors[0])))
        print("test vectors shape: %d * %d" % (len(test_vectors), len(test_vectors[0])))


if __name__ == "__main__":
    unittest.main()