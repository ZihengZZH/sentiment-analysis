import unittest
import random
from src.embeddings.bow import BagOfWords
from src.utils.data import dataLoader



class BoWFeatureTest(unittest.TestCase):
    def test_bow(self):
        loader = dataLoader()
        data = loader.load_data('cam_data')
        
        train_data = data[:-100]
        test_data = data[-100:]
        bow_new = BagOfWords(ngram='unigram', 
                            docs_train=train_data, 
                            docs_test=test_data, 
                            dataset='cam_data')
        bow_new.save_bow()


if __name__ == "__main__":
    unittest.main()
