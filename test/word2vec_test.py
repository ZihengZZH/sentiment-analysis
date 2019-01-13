import unittest
import re
import src.word2vec as word2vec

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.word2vec_test
'''

class word2vecTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.sample_text = "He is the king. The king is royal. She is the royal queen. She is the queen"
        self.sample_words = re.split('\W+', self.sample_text.lower())
        self.sample_sentences = self.sample_text.lower().split('.')

    # method called immediately after the test method has been called and the result recoreded
    def tearDown(self):
        pass

    def test_word2vec_naive(self):
        # input_texts = []
        # for _ in range(3):
        #     input_texts.append(self.sample_words)
        # print("\nbegin the test of word2vec implementation in TensorFlow ...")
        # print("\nget the vocabulary/corpus of the sample text ...")
        # vocab, vocab_size, word2int, int2word = word2vec.get_vocab(input_texts)
        # print("\ngenerate the training data ...")
        # X_train, y_train = word2vec.prepare_train_data(self.sample_sentences, vocab_size, word2int)
        # print("\nbuild and train the skip-gram models ...")
        # vectors = word2vec.run_tensorflow_model(X_train, y_train, vocab_size, verbose=False)
        # print(int2word[word2vec.find_closest_vector(word2int['king'], vectors)])
        # print(int2word[word2vec.find_closest_vector(word2int['queen'], vectors)])
        # print(int2word[word2vec.find_closest_vector(word2int['royal'], vectors)])
        # word2vec.get_TSNE(vectors, word2int, vocab)
        pass

    def test_gensim_implement(self):
        word2vec.gensim_implement('./dataset/sample.txt')



        

