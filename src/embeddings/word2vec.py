import os
import math
import random
import zipfile
import datetime
import subprocess
import collections
import progressbar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from gensim.models import word2vec


WINDOW_SIZE = 2
EMBEDDING_DIM = 7
N_INTERS = 10000


# return the vocabulary for all movie reviews
def get_vocab(input_texts):
    # para input_texts: list(list(str))
    vocab = list()
    word2int, int2word = dict(), dict()
    bar = progressbar.ProgressBar()

    for t in bar(range(len(input_texts))):
        texts = input_texts[t]
        for word in texts:
            if word not in vocab:
                vocab.append(word)
    
    vocab_size = len(vocab)
    for i, word in enumerate(vocab):
        word2int[word] = i
        int2word[i] = word
    return vocab, vocab_size, word2int, int2word


# return the one-hot vector
def to_one_hot_vector(data_point_index, vocab_size):
    # para data_point_index: integer
    # para vocab_size: integer
    one_hot = np.zeros(vocab_size)
    one_hot[data_point_index] = 1
    return one_hot


# return the training data
def prepare_train_data(sentences, vocab_size, word2int):
    # para sentences: list(str)
    # para vocab_size: integer
    # para word2int: dict
    data = list()
    x_train, y_train = list(), list()
    bar = progressbar.ProgressBar()

    for i in bar(range(len(sentences))):
        sentence = sentences[i].split()
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
                if nb_word != word:
                    data.append([word, nb_word])
    
    for data_word in data:
        x_train.append(to_one_hot_vector(word2int[ data_word[0] ], vocab_size))
        y_train.append(to_one_hot_vector(word2int[ data_word[1] ], vocab_size))

    return np.array(x_train), np.array(y_train)


'''one_hot_vector --> embedded repre. --> predicted_neighbour_prob'''

def run_tensorflow_model(X_train, y_train, vocab_size, verbose=False):
    # para X_train:
    # para y_train:
    # para vocab_size: 
    X_data = tf.placeholder(tf.float32, shape=(None, vocab_size))
    y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

    W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) # bias

    hidden_repre = tf.add(tf.matmul(X_data, W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
    b2 = tf.Variable(tf.random_normal([vocab_size]))
    
    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_repre, W2), b2))

    sess = tf.Session()
    
    init = tf.global_variables_initializer()

    sess.run(init)

    # define the loss function
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

    # define the training step
    train_step = tf.train.GradientDescentOptimizer(.1).minimize(cross_entropy_loss)

    bar = progressbar.ProgressBar()

    for i in bar(range(N_INTERS)):
        sess.run(train_step, feed_dict={X_data: X_train, y_label: y_train})
        if i % 100 == 0 and verbose:
            print("loss is : ", sess.run(cross_entropy_loss, feed_dict={X_data: X_train, y_label: y_train}))
    
    if verbose:
        print(sess.run(W1))
        print("-"*20)
        print(sess.run(b1))
        print("-"*20)
    
    vectors = sess.run(W1 + b1)
    return vectors


def euclidean_dist(vector1, vector2):
    return np.sqrt(np.sum(vector1 - vector2) ** 2)


def find_closest_vector(word_index, vectors):
    # para word_index:
    # para vectors:
    min_dist = 10000
    min_index = -1

    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    
    return index


def get_TSNE(vectors, word2int, vocab):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    vectors = model.fit_transform(vectors)
    normalizer = preprocessing.Normalizer()
    vectors_norm = normalizer.fit_transform(vectors, 'l2')

    fig, ax = plt.subplots()
    for word in vocab:
        x_coord = vectors_norm[word2int[word]][0]
        y_coord = vectors_norm[word2int[word]][1]
        print("%s, (%f, %f)" % (word, x_coord, y_coord))
        ax.annotate(word, (x_coord, y_coord))
    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    plt.show()


def gensim_implement(text):
    sentences = word2vec.Text8Corpus(text)
    model = word2vec.Word2Vec(sentences)
    for e in model.most_similar(positive=['king'], topn=1):
        print(e[0], e[1])
    
    print(model.similarity('she', 'queen'))
