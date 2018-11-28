import numpy as np


# return the vector of labels (neg/0 or pos/1)
def get_class_vec(sentiment, length):
    if sentiment == 'neg':
        return np.zeros(length)
    else:
        return np.ones(length)


# return the vocabulary for all moive reviews
def get_vocab(input_texts):
    # para input_texts: list(list(tuple(str,str)))
    vocab = set()
    for texts in input_texts:
        for text in texts:
            vocab.add(text)  # union of two sets
    return list(vocab)


# return the matrix in which each row is the feature vector
def bag_words2vec_unigram(vocab, input_texts, freq):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with freq or pres
    vec2mat = []
    for text in input_texts:
        vec2mat.append(words2vec_unigram(vocab, text, freq))
    return np.array(vec2mat)


# return the feature vector
def words2vec_unigram(vocab, input_text, freq):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with freq or pres
    vec_unigram = [0]*len(vocab)  # vector for each review
    for word in input_text:
        if word in vocab:
            if freq:
                vec_unigram[vocab.index(word)] += 1  # frequency
            else:
                vec_unigram[vocab.index(word)] = 1   # presence
    return vec_unigram


def bag_words2vec_bigram(vocab, input_texts, freq):
    # para texts: data & tags read from the text file
    # return para: list of dict
    # return para: dict or copus with freq or pres
    vec2mat = []
    #for text in input_texts:



def words2vec_bigram(vocab, input_text, freq):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with freq or pres
    vec = [0]


def visual_unigram(corpus, vec2mat):
    print("--"*20)
    print("word&tag\t\t\tcount")
    for i in range(len(corpus)):
        print(str(corpus[i]).ljust(32), vec2mat[0][i])


def visual_bigram(bigrams):
    return 0
