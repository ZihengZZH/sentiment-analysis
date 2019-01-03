import numpy as np

import progressbar
from multiprocessing import cpu_count, Pool
from functools import partial


# return the vector of labels (neg/0 or pos/1)
def get_class_vec(sentiment, length):
    if sentiment == 'neg':
        return np.zeros(length)
    else:
        return np.ones(length)


# return the vocabulary for all movie reviews
def get_vocab(input_texts, cutoff_threshold=0):
    # para input_texts: list(list(str))
    # DATA_TAG: list(list(tuple(str,str)))
    # para cutoff_threshold: predetermined: 10546/63331 (>9)
    # return para: # feature
    freq_dict, vocab = dict(), list()

    bar = progressbar.ProgressBar()
    
    for t in bar(range(len(input_texts))):
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


# return the vocabulart for all movie reviews (pair)
def get_vocab_bigram(input_texts, cutoff_threshold=0):
    # para input_texts: list(list(str))
    # DATA_TAG: list(list(tuple(str,str)))
    # para cutoff_threshold: predetermined: 10918/502596 (>14)
    # return para: # feature list(tuple(tuple(str,str)))
    freq_dict, vocab = dict(), list()

    bar = progressbar.ProgressBar()

    for t in bar(range(len(input_texts))):
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
    # return dict of vocabulary and counts
    vocab = dict()
    for texts in input_texts:
        for text in texts:
            if text in vocab:
                vocab[text] += 1
            else:
                vocab[text] = 1
    return vocab


# return the matrix in which each row is a feature vector
def bag_words2vec_unigram(vocab, input_texts):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with freq
    vec2mat = np.zeros((len(input_texts), len(vocab)))

    # bar = progressbar.ProgressBar()
    NUM_PROCESS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCESS)
    
    vec2mat = np.array(list(pool.map(partial(words2vec_unigram, vocab), input_texts)))

    return vec2mat


def bag_words2vec_unigram_naive(vocab, input_texts):
    vec2mat = []
    for text in input_texts:
        vec2mat.append(words2vec_unigram(vocab, text))
    return np.array(vec2mat)


# return the feature vector
def words2vec_unigram(vocab, input_text):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with freq
    # return type: list[integer]
    vec_unigram = [0]*len(vocab)  # vector for each review
    for word in input_text:
        if word in vocab:
            vec_unigram[vocab.index(word)] += 1  # frequency
    return vec_unigram


# return the matrix in which each row is a feature vector (bigram)
def bag_words2vec_bigram(vocab, input_texts):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # para texts: data & tags read from the text file
    # return para: dict or copus with freq
    vec2mat = np.zeros((len(input_texts), len(vocab)))

    # bar = progressbar.ProgressBar()
    NUM_PROCESS = cpu_count() * 3
    pool = Pool(processes=NUM_PROCESS)
   
    vec2mat = np.array(list(pool.map(partial(words2vec_bigram, vocab), input_texts)))
    
    return vec2mat


def bag_words2vec_bigram_naive(vocab, input_texts):
    vec2mat = []
    for text in input_texts:
        vec2mat.append(words2vec_bigram(vocab, text))
    return np.array(vec2mat)


# return the feature vector (bigram)
def words2vec_bigram(vocab, input_text):
    # para input_text: data & tags read from the text file
    # para input_text: list(tuple(str,str))
    # return para: dict or copus with freq
    # return type: list[integer]
    vec_bigram = [0]*len(vocab)
    for i in range(len(input_text)-1):
        if (input_text[i], input_text[i+1]) in vocab:
            vec_bigram[vocab.index((input_text[i], input_text[i+1]))] += 1
    return vec_bigram


def visual_matrix(vocab, vec2mat):
    # just visualise first text
    print("--"*20)
    print("word&tag\t\t\tcount")
    for i in range(len(vocab)):
        print(str(vocab[i]).ljust(32), vec2mat[0][i])


def visual_matrix_bigram(vocab, vec2mat):
    # just visualise first text
    print("--"*20)
    print("word&tag\tword&tag\t\t\tcount")
    for i in range(len(vocab)):
        print(vocab[i][0], vocab[i][1], str(vec2mat[0][i]).rjust(30))


def concatenate_feat(vec2mat_uni, vec2mat_bi):
    vec2mat = list()
    for i in range(len(vec2mat_bi)):
        vec2mat.append(vec2mat_uni[i] + vec2mat_bi[i])
    return vec2mat
