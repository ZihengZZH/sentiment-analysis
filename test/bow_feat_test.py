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
        # Two teen couples Two teen couples party They into !
        # One of guys to teen in life girlfriend accident !
        self.sample_text.append([('Two', 'CD'), ('teen', 'JJ'), ('couples', 'NNS'), ('Two', 'CD'), (
            'teen', 'JJ'), ('couples', 'NNS'), ('party', 'NN'), ('They', 'PRP'), ('into', 'IN'), ('!', '!')])
        self.sample_text.append([('One', 'CD'), ('of', 'IN'), ('guys', 'NNS'), ('to', 'TO'), (
            'teen', 'JJ'), ('in', 'IN'), ('life', 'NN'), ('girlfriend', 'NN'), ('accident', 'NN'), ('!', '!')])
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_freq_cutoff(self):
        neg_texts = text.read_data_from_file('neg')
        pos_texts = text.read_data_from_file('pos')
        vocab_unigram = feat.get_vocab(neg_texts+pos_texts,9)
        vocab_bigram = feat.get_vocab_bigram(neg_texts+pos_texts,14)
        print(len(vocab_unigram))
        print(len(vocab_bigram))
    

    def test_bag_words2vec_unigram(self):
        # check algorithms on the sample reviews
        sample_vocab = feat.get_vocab(self.sample_text)
        mat_feat = feat.bag_words2vec_unigram(sample_vocab, self.sample_text)
        mat_feat_naive = feat.bag_words2vec_unigram_naive(sample_vocab, self.sample_text)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 2)])
        assert mat_feat_naive.all() == mat_feat.all()


    def test_bag_words2vec_unigram_real(self):
        # take only 10 reviews into test part
        texts = text.read_data_from_file('neg')[:10]
        sample_vocab = feat.get_vocab(texts, 9)
        mat_feat = feat.bag_words2vec_unigram(sample_vocab, texts)
        mat_feat_naive = feat.bag_words2vec_bigram_naive(sample_vocab, texts)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 10)])
        assert mat_feat_naive.all() == mat_feat.all()
        feat.visual_matrix(sample_vocab, mat_feat)

    def test_bag_words2vec_bigram(self):
        # check algorithms on the sample reviews
        sample_vocab = feat.get_vocab_bigram(self.sample_text)
        mat_feat = feat.bag_words2vec_bigram(sample_vocab, self.sample_text)
        mat_feat_naive = feat.bag_words2vec_bigram_naive(sample_vocab, self.sample_text)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 2)])
        assert mat_feat_naive.all() == mat_feat.all()

    def test_bag_words2vec_bigram_real(self):
        # take only 10 reviews into test part
        texts = text.read_data_from_file('neg')[:10]
        sample_vocab = feat.get_vocab_bigram(texts, 14)
        mat_feat = feat.bag_words2vec_bigram(sample_vocab, texts)
        mat_feat_naive = feat.bag_words2vec_bigram_naive(sample_vocab, texts)
        assert len(sample_vocab) == len(mat_feat[random.randrange(0, 10)])
        assert mat_feat_naive.all() == mat_feat.all()


if __name__ == "__main__":
    unittest.main()
