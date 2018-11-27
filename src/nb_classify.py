import numpy as np
import math


def train_nb_classifier(train_mat, train_c):
    # para train_mat: matrix of data & tags
    # para train_c: sentiment (neg/0 or pos/1)
    no_train_review = len(train_mat)
    no_words = len(train_mat[0])
    prob_sentiment = sum(train_c) / float(no_train_review)  # probability 0.5
    #
    prob_neg_num = np.ones(no_words)
    #
    prob_pos_num = np.ones(no_words)
    prob_neg_denom = 2.0
    prob_pos_denom = 2.0
    # iteration on every training review
    for i in range(no_train_review):
        if train_c[i] == 0:
            prob_neg_num += train_mat[i]
            prob_neg_denom += sum(train_mat[i])
        else:
            prob_pos_num += train_mat[i]
            prob_pos_denom += sum(train_mat[i])
    # prob vector for negative reviews
    prob_neg_vec = np.log(prob_neg_num/prob_neg_denom)
    # prob vector for positive reviews
    prob_pos_vec = np.log(prob_pos_num/prob_pos_denom)
    return prob_neg_vec, prob_pos_vec, prob_sentiment


def test_nb_classifier(test_vec, prob_neg_vec, prob_pos_vec, prob_class):
    prob_neg = sum(test_vec*prob_neg_vec) + np.log(1.0-prob_class)
    prob_pos = sum(test_vec*prob_pos_vec) + np.log(prob_class)
    if prob_neg > prob_pos:
        return 0
    else:
        return 1
