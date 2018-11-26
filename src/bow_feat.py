import numpy as np


# return the vector of labels (neg/0 or pos/1)
def get_class_vec(sentiment, length):
    if sentiment == 'neg':
        return np.zeros(length)
    else:
        return np.ones(length)


# return the corpus for all moive reviews
def corpus_list(input_texts):
    # para input_texts: list(list(tuple(str,str)))
    corpus = set()
    for texts in input_texts:
        for text in texts:
            corpus.add(text)  # union of two sets
    return list(corpus)


# return the matrix in which each row is the feature vector
# note that presences are applied
def bag_words2vec_unigram(corpus, input_texts):
    # para input_texts: data & tags read from the text file
    # para input_texts: list(list(tuple(str,str)))
    # return para: dict or copus with presence
    vec2mat = []
    for text in input_texts:
        vec_unigram = [0]*len(corpus)  # vector for each review
        for word in text:
            if word in corpus:
                vec_unigram[corpus.index(word)] += 1  # presence
        vec2mat.append(vec_unigram)
    return np.array(vec2mat)


def set_words2vec_bigram(texts):
    # para texts: data & tags read from the text file
    # return para: list of dict
    return 0


def visual_unigram(corpus, vec2mat):
    print("--"*20)
    print("word&tag\t\t\tcount")
    for i in range(len(corpus)):
        print(str(corpus[i]).ljust(32), vec2mat[0][i])


def visual_bigram(bigrams):
    return 0
