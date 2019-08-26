import unittest
import random
from src.embeddings.bow import BagOfWords
from src.utils.data import load_data



class BoWFeatureTest(unittest.TestCase):
    def test_bow(self):
        data = load_data('cam_data')
        
        train_data = data[:-100]
        test_data = data[-100:]
        bow_new = BagOfWords('unigram', train_data, test_data, 'cam_data')
        bow_new.save_bow()


if __name__ == "__main__":
    unittest.main()
