import unittest
import random
import src.bow_feat as feat
import src.text as text

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.bow_feat_test
'''


class BoWFeatureTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        random.seed(0)
        self.sample_text = list()
        self.sample_text.append([('Two', 'CD'), ('teen', 'JJ'), ('couples', 'NNS'), ('Two', 'CD'), (
            'teen', 'JJ'), ('couples', 'NNS'), ('party', 'NN'), ('They', 'PRP'), ('into', 'IN'), ('!', '!')])
        self.sample_text.append([('One', 'CD'), ('of', 'IN'), ('guys', 'NNS'), ('to', 'TO'), (
            'teen', 'JJ'), ('in', 'IN'), ('life', 'NN'), ('girlfriend', 'NN'), ('accident', 'NN'), ('!', '!')])

    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_bag_words2vec_unigram(self):
        # check algorithms on the sample reviews
        sample_vocab = feat.get_vocab(self.sample_text)
        mat_feat = feat.bag_words2vec_unigram(sample_vocab, self.sample_text, False)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 2)])
        feat.visual_unigram(sample_vocab, mat_feat)

    def test_bag_words2vec_unigram_real(self):
        # take only 10 reviews into test part
        texts = text.read_data_tag_from_file('neg')[:10]
        sample_vocab = feat.get_vocab(texts)
        mat_feat = feat.bag_words2vec_unigram(sample_vocab, texts, False)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 10)])
        feat.visual_unigram(sample_vocab, mat_feat)


if __name__ == "__main__":
    unittest.main()
