import os
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from functools import partial
from src.utils.display import print_new


def get_class_vec(sentiment, length):
    """return the vector of labels (neg/0 or pos/1)
    """
    if sentiment == 'neg':
        return np.zeros(length)
    else:
        return np.ones(length)


def get_vocab_unigram(input_texts, cutoff_threshold=0):
    """return the vocabulary for all documents (unigrams)
    --
    # para input_texts: list(list(str))
    # DATA_TAG: list(list(tuple(str,str)))
    # para cutoff_threshold: predetermined: 10546/63331 (>9)
    # return para: vocabulary list of all reviews
    # return type: list(str)
    """
    freq_dict, vocab = dict(), list()
    for t in tqdm(range(len(input_texts))):
        texts = input_texts[t]
        for text in texts:
            if text in freq_dict:
                freq_dict[text] += 1
            else:
                freq_dict[text] = 1
    for key, val in freq_dict.items():
        if val > cutoff_threshold:
            vocab.append(key)
    return vocab


def get_vocab_bigram(input_texts, cutoff_threshold=0):
    """return the vocabulary for all documents (bigrams)
    --
    # para input_texts: list(list(str))
    # DATA_TAG: list(list(tuple(str,str)))
    # para cutoff_threshold: predetermined: 10918/502596 (>14)
    # return para: vocabulary list of all reviews 
    # return type: list(str)
    """
    freq_dict, vocab = dict(), list()

    for t in tqdm(range(len(input_texts))):
        texts = input_texts[t]
        for i in range(len(texts)-1):
            if (texts[i], texts[i+1]) in freq_dict:
                freq_dict[(texts[i], texts[i+1])] += 1
            else:
                freq_dict[(texts[i], texts[i+1])] = 1
    for key, val in freq_dict.items():
        if val > cutoff_threshold:
            vocab.append(key)
    return vocab


def get_vocab_count(input_texts):
    """return dict of vocabulary and counts
    """
    vocab = dict()
    for texts in input_texts:
        for text in texts:
            if text in vocab:
                vocab[text] += 1
            else:
                vocab[text] = 1
    return vocab


def bag_words2vec_unigram(vocab, input_texts):
    """return matrix of unigrams
    --
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or corpus with freq
    """
    vec2mat = np.zeros((len(input_texts), len(vocab)))
    NUM_PROCESS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCESS)
    vec2mat = np.array(list(pool.map(partial(words2vec_unigram, vocab), input_texts)))
    return vec2mat


def bag_words2vec_unigram_naive(vocab, input_texts):
    """return matrix of unigrams (naive implementation)
    """
    vec2mat = []
    for text in input_texts:
        vec2mat.append(words2vec_unigram(vocab, text))
    return np.array(vec2mat)


def words2vec_unigram(vocab, input_text):
    """return feature vectors for unigrams
    --
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or corpus with freq
    # return type: list[integer]
    """
    vec_unigram = [0]*len(vocab)  # vector for each review
    for word in input_text:
        if word in vocab:
            vec_unigram[vocab.index(word)] += 1  # frequency
    return vec_unigram


def bag_words2vec_bigram(vocab, input_texts):
    """return matrix of bigrams
    --
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # para texts: data & tags read from the text file
    # return para: dict or corpus with freq
    """
    vec2mat = np.zeros((len(input_texts), len(vocab)))
    NUM_PROCESS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCESS)
    vec2mat = np.array(list(pool.map(partial(words2vec_bigram, vocab), input_texts)))
    return vec2mat


def bag_words2vec_bigram_naive(vocab, input_texts):
    """return matrix of bigrams (naive implementation)
    """
    vec2mat = []
    for text in input_texts:
        vec2mat.append(words2vec_bigram(vocab, text))
    return np.array(vec2mat)


def words2vec_bigram(vocab, input_text):
    """return feature vectors for bigrams
    --
    # para input_text: data & tags read from the text file
    # para input_text: list(tuple(str,str))
    # return para: dict or corpus with freq
    # return type: list[integer]
    """
    vec_bigram = [0]*len(vocab)  # vector for each review
    for i in range(len(input_text)-1):
        if (input_text[i], input_text[i+1]) in vocab:
            vec_bigram[vocab.index((input_text[i], input_text[i+1]))] += 1  # frequency
    return vec_bigram


def visual_matrix_unigram(vocab, vec2mat):
    """just visualize first review of unigrams
    """
    print("--"*20)
    print("word&tag\t\t\tcount")
    for i in range(len(vocab)):
        print(str(vocab[i]).ljust(32), vec2mat[0][i])


def visual_matrix_bigram(vocab, vec2mat):
    """just visualize first review of bigrams
    """
    print("--"*20)
    print("word&tag\tword&tag\t\t\tcount")
    for i in range(len(vocab)):
        print(vocab[i][0], vocab[i][1], str(vec2mat[0][i]).rjust(30))


def concatenate_feat(vec2mat_unigram, vec2mat_bigram):
    """concatenate unigrams and bigrams
    """
    assert len(vec2mat_unigram) == len(vec2mat_bigram), "LENGTH MISMATCH"
    vec2mat = list()
    for i in tqdm(range(len(vec2mat_bigram))):
        vec2mat.append(vec2mat_unigram[i] + vec2mat_bigram[i])
    return np.array(vec2mat)


class BagOfWords(object):
    """
    Bag-of-Words features (Uni/Bi-grams)
    --
    # para ngram: 'unigram' / 'bigram' / 'concat'
    # para docs_train: train documents
    # para docs_test: test documents
    # para data_path: data location
    """
    def __init__(self, ngram, docs_train, docs_test, dataset):
        self.config = json.load(open('./config.json', 'r', encoding='utf-8'))
        self.ngram = ngram
        self.docs_train = docs_train
        self.docs_test = docs_test
        self.data_path = os.path.join(self.config['data_path'], 
                                    self.config['dataset'][dataset])
        self.vocab = None
        self.X_train = None
        self.X_test = None
        self.unigram_cutoff = self.config['bow']['unigram_cutoff']
        self.bigram_cutoff = self.config['bow']['bigram_cutoff']
    
    def save_bow(self):
        """save bag-of-words representations to external files
        """
        # get unigrams
        if self.ngram == 'unigram':
            self.vocab = get_vocab_unigram(self.docs_train, cutoff_threshold=self.unigram_cutoff)
            self.X_train = bag_words2vec_unigram(self.vocab, self.docs_train)
            self.X_test = bag_words2vec_unigram(self.vocab, self.docs_test)
        # get bigrams
        elif self.ngram == 'bigram':
            self.vocab = get_vocab_bigram(self.docs_train, cutoff_threshold=self.bigram_cutoff)
            self.X_train = bag_words2vec_bigram(self.vocab, self.docs_train)
            self.X_test = bag_words2vec_bigram(self.vocab, self.docs_test)
        # get concatenation of unigrams and bigrams
        else:
            self.vocab = get_vocab_unigram(self.docs_train, cutoff_threshold=self.unigram_cutoff) + \
                        get_vocab_bigram(self.docs_train, cutoff_threshold=self.bigram_cutoff)
            X_train_unigram = bag_words2vec_unigram(self.vocab, self.docs_train)
            X_train_bigram = bag_words2vec_bigram(self.vocab, self.docs_train)
            X_test_unigram = bag_words2vec_unigram(self.vocab, self.docs_test)
            X_test_bigram = bag_words2vec_bigram(self.vocab, self.docs_test)
            self.X_train = concatenate_feat(X_train_unigram, X_train_bigram)
            self.X_test = concatenate_feat(X_test_unigram, X_test_bigram)

        np.save(os.path.join(self.data_path, 'vocabulary'), self.vocab)
        np.save(os.path.join(self.data_path, 'X_train_bow'), self.X_train)
        np.save(os.path.join(self.data_path, 'X_test_bow'), self.X_test)

    def load_bow(self):
        """load bag-of-words representations from external files
        """
        try:
            self.vocab = np.load(os.path.join(self.data_path, 'vocabulary.npy'))
            self.X_train = np.load(os.path.join(self.data_path, 'X_train_bow.npy'))
            self.X_test = np.load(os.path.join(self.data_path, 'X_test_bow.npy'))
        except:
            print_new("NO BOW FEATURES AVAILABLE")

    def visualize(self):
        """perform some visualization
        """
        if self.ngram == 'unigram':
            visual_matrix_unigram(self.vocab, self.X_train)
        elif self.ngram == 'bigram':
            visual_matrix_bigram(self.vocab, self.X_train)
