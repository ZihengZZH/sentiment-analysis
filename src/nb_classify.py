import numpy as np
from scipy import stats
import math
from . import text
from . import bow_feat as feat


# train/test partition / cross-validation
def partition(neg_reviews, pos_reviews, test=False):
    # return para: train size of each class
    if test:
        train_size, test_size = 200, 50
        neg_reviews_train, neg_reviews_test = neg_reviews[:
                                                          train_size], neg_reviews[train_size:train_size+test_size]
        pos_reviews_train, pos_reviews_test = pos_reviews[:
                                                          train_size], pos_reviews[train_size:train_size+test_size]
    else:
        train_size, test_size = 900, 100
        neg_reviews_train, neg_reviews_test = neg_reviews[:
                                                          train_size], neg_reviews[train_size:]
        pos_reviews_train, pos_reviews_test = pos_reviews[:
                                                          train_size], pos_reviews[train_size:]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train
    reviews_test = [neg_reviews_test, pos_reviews_test]
    return train_size, test_size, reviews_train, reviews_test


def train_nb_classifier(train_mat, train_c, smooth_type, feat_type):
    # para train_mat: matrix of data & tags
    # para train_c: R^2 sentiment (neg/0 or pos/1)
    # return para: prob_vec: R^2 vec of conditional prob
    # return para: prior_sentiment
    if (smooth_type != 'laplace') and (smooth_type != 'None'):
        return
    if (feat_type != 'unigram') and (feat_type != 'bigram'):
        return 
    
    no_train_review = len(train_mat)
    no_words = len(train_mat[0])
    prior_sentiment = sum(train_c) / float(no_train_review)  # prob 0.5

    # check whether smoothing or not
    # numerator Tct
    prob_neg_num = np.zeros(no_words)
    prob_pos_num = np.zeros(no_words)
    # denominator sum(Tct)
    prob_neg_denom = .0
    prob_pos_denom = .0

    k = 0.0
    if smooth_type == 'laplace':
        k = 2.0  # constant for all words

    # iteration on every training review
    for i in range(no_train_review):
        if train_c[i] == 0:
            prob_neg_num += train_mat[i]
            prob_neg_denom += sum(train_mat[i])
        else:
            prob_pos_num += train_mat[i]
            prob_pos_denom += sum(train_mat[i])
    # prob vector for negative reviews P(fi|0)
    prob_neg_vec = (prob_neg_num+k)/(prob_neg_denom+k)
    # prob vector for positive reviews P(fi|1)
    prob_pos_vec = (prob_pos_num+k)/(prob_pos_denom+k)

    return prob_neg_vec, prob_pos_vec, prior_sentiment


def test_nb_classifier(test_vec, prob_neg_vec, prob_pos_vec, prior_class):
    # para prob_neg_vec: P(fi|0)
    # para prob_pos_vec: P(fi|1)
    # para prob_class: P(c=0) or P(c=1) (equal in this project)
    # conditional prob of features already log
    prob_neg = sum(test_vec*np.log(prob_neg_vec)) + np.log(1.0-prior_class)
    prob_pos = sum(test_vec*np.log(prob_pos_vec)) + np.log(prior_class)
    # binary classification argmax
    if prob_neg > prob_pos:
        # predict a negative review
        return 0
    else:
        # predict a positive review
        return 1


def nb_classifier(feat_type, smoothing, test=False):
    # para smoothing: the type of smoothing
    # para test: test case or not

    print("\nreading reviews from files ...")
    # read all neg and pos reviews
    neg_reviews = text.read_data_tag_from_file('neg')
    pos_reviews = text.read_data_tag_from_file('pos')

    print("\ntrain/test partitioning ...")
    train_size, test_size, reviews_train, reviews_test = partition(
        neg_reviews, pos_reviews, test)

    print("\nfinding the vocabulary for the classifier ...")
    # full vocabulary for the training reviews (frequency cutoff implemented)
    full_vocab = feat.get_vocab(
        reviews_train, cutoff_threshold=9) if feat_type == 'unigram' else feat.get_vocab_bigram(reviews_train, cutoff_threshold=14)
    print("\n#features is ", len(full_vocab))

    print("\ngenerating the training matrix ...")
    # training matrix of data and tags
    train_matrix = feat.bag_words2vec_unigram(
        full_vocab, reviews_train) if feat_type == 'unigram' else feat.bag_words2vec_bigram(full_vocab, reviews_train)
    if test:
        print('\ndescription of training matrix', stats.describe(train_matrix))

    # training vectors of labels
    train_class_vector = np.hstack(
        (feat.get_class_vec('neg', train_size), feat.get_class_vec('pos', train_size)))

    print("\ntraining the Naive Bayes classifier ...")
    # train the Naive Bayes classifier
    prob_neg_vec, prob_pos_vec, prob_sentiment = train_nb_classifier(
        train_matrix, train_class_vector, smoothing, feat_type)
    print("prob vector on neg reviews", prob_neg_vec, "\nprob vector on pos reviews",
          prob_pos_vec, "\nprob of sentiment", prob_sentiment)

    print("\ntesting the Naive Bayes classifier ...")
    # test the classifier with another review
    i, neg_score, pos_score = 0, 0, 0
    for review_test in reviews_test[0]:
        test_vec = feat.words2vec_unigram(
            full_vocab, review_test) if feat_type == 'unigram' else feat.words2vec_bigram(full_vocab, review_test)
        test_result = test_nb_classifier(
            test_vec, prob_neg_vec, prob_pos_vec, prob_sentiment)
        neg_score += test_result
        i += 1
        print("Test sample %d \nThis review is %d review, 0 for neg" %
              (i, test_result))
    for review_test in reviews_test[1]:
        test_vec = feat.words2vec_unigram(
            full_vocab, review_test) if feat_type == 'unigram' else feat.words2vec_bigram(full_vocab, review_test)
        test_result = test_nb_classifier(
            test_vec, prob_neg_vec, prob_pos_vec, prob_sentiment)
        pos_score += test_result
        print("Test sample %d \nThis review is %d review, 1 for pos" %
              (i, test_result))
        i += 1

    # print overall accuracy
    print("overall accuracy for negative reviews", 1-(neg_score/test_size))
    print("overall accuracy for positive reviews", (pos_score/test_size))

    # write results to text file
    f = open('./results.txt', 'a+', encoding='utf-8')
    f.write("\nfeature: %s\ttraining size: %d\tsmooth: %s\tneg_accuracy: %f\tpos_accuracy: %f" % (
        feat_type, train_size, 'laplace', 1-(neg_score/test_size), (pos_score/test_size)))
    f.close()
    print("\written to files ...")
