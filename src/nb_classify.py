import numpy as np
from scipy import stats
import math
from . import text
from . import bow_feat as feat
from . import sign_test as st


def n_fold_cons(no_fold, length_data):
    # in case data cannot separated evenly
    length_split = int(length_data / no_fold)
    test_range = list()
    for i in range(no_fold):
        test_range.append([length_split*i, length_split*(i+1)])
    return test_range


# Round-robin splitting mod 10
def n_fold_RR(no_fold, length_data):
    mod = no_fold # basically the same
    test_splits = list()
    length_split = int(length_data / no_fold)
    for i in range(no_fold):
        test_split = list()
        for j in range(length_split):
            test_split.append(i+mod*j)
        test_splits.append(test_split)
    return test_splits


# train/test partition / cross-validation
def partition(neg_reviews, pos_reviews, test):
    # return para: train size of each class
    if test:
        train_size, test_size = 100, 50
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
    reviews_train = neg_reviews_train + pos_reviews_train  # dimension: 1
    reviews_test = neg_reviews_test + pos_reviews_test  # dimension: 1
    return train_size, test_size, reviews_train, reviews_test


def prepare_data(test):
    # read all neg and pos reviews
    neg_reviews = text.read_data_tag_from_file('neg')
    pos_reviews = text.read_data_tag_from_file('pos')

    print("\ntrain/test partitioning ...")
    train_size, test_size, reviews_train, reviews_test = partition(
        neg_reviews, pos_reviews, test)
    
    return train_size, test_size, reviews_train, reviews_test


def prepare_data_tenfold(neg_reviews, pos_reviews, test_range):
    # para test_range: list[start:end]
    train_size, test_size = 900, 100
    [start_point, end_point] = test_range
    neg_reviews_train = neg_reviews[:start_point] + neg_reviews[end_point:]
    neg_reviews_test = neg_reviews[start_point:end_point]
    pos_reviews_train = pos_reviews[:start_point] + pos_reviews[end_point:]
    pos_reviews_test = pos_reviews[start_point:end_point]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train  # dimension: 1
    reviews_test = neg_reviews_test + pos_reviews_test  # dimension: 1
    return train_size, test_size, reviews_train, reviews_test


def prepare_data_roundrobin(neg_reviews, pos_reviews, test_range):
    # para test_range: list[index]
    train_size, test_size = 900, 100
    neg_reviews_train, neg_reviews_test, pos_reviews_train, pos_reviews_test = [], [], [], []
    for ele in test_range:
        neg_reviews_train += neg_reviews[ele]
        pos_reviews_train += pos_reviews[ele]
        
    train_range = list(set(range(1000)) - set(test_range))
    for ele in train_range:
        neg_reviews_test += neg_reviews[ele]
        pos_reviews_test += pos_reviews[ele]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train  # dimension: 1
    reviews_test = neg_reviews_test + pos_reviews_test  # dimension: 1
    return train_size, test_size, reviews_train, reviews_test


def train_nb_classifier(train_mat, train_c, smooth_type):
    # para train_mat: matrix of data & tags
    # para train_c: R^2 sentiment (neg/0 or pos/1)
    # return para: prob_vec: R^2 vec of conditional prob
    # return para: prior_sentiment
    if (smooth_type != 'laplace') and (smooth_type != 'None'):
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

    if smooth_type == 'laplace':
        k = 2.0  # constant for all words
    else:
        k = 0.0

    # iteration on every training review
    for i in range(no_train_review):
        if train_c[i] == 0:
            prob_neg_num += train_mat[i]
            prob_neg_denom += (sum(train_mat[i])+k)
        else:
            prob_pos_num += train_mat[i]
            prob_pos_denom += (sum(train_mat[i])+k)
    # prob vector for negative reviews P(fi|0)
    prob_neg_vec = (prob_neg_num+k)/(prob_neg_denom)
    # prob vector for positive reviews P(fi|1)
    prob_pos_vec = (prob_pos_num+k)/(prob_pos_denom)

    return prob_neg_vec, prob_pos_vec, prior_sentiment


def test_nb_classifier(test_vec, prob_neg_vec, prob_pos_vec, prior_class):
    # para prob_neg_vec: P(fi|0)
    # para prob_pos_vec: P(fi|1)
    # para prob_class: P(c=0) or P(c=1) (equal in this project)
    # avoid the nan in np.log(0) calculation
    prob_neg_log = np.log(prob_neg_vec)
    prob_pos_log = np.log(prob_pos_vec)
    for i in range(len(prob_neg_log)):
        if np.isinf(prob_neg_log[i]):
            prob_neg_log[i] = 0.0
        if np.isinf(prob_pos_log[i]):
            prob_pos_log[i] = 0.0

    prob_neg = sum(test_vec*prob_neg_log) + np.log(1.0-prior_class)
    prob_pos = sum(test_vec*prob_pos_log) + np.log(prior_class)
    # binary classification argmax
    if prob_neg > prob_pos:
        # predict a negative review
        return 0
    else:
        # predict a positive review
        return 1


def nb_classifier_train_test(train_size, test_size, reviews_train, reviews_test, feat_type, smoothing, test):
    print("\nfinding the vocabulary for the classifier ...")
    # full vocabulary for the training reviews (frequency cutoff implemented)
    if feat_type == 'unigram':
        full_vocab = feat.get_vocab(reviews_train, cutoff_threshold=9)
    elif feat_type == 'bigram':
        full_vocab = feat.get_vocab_bigram(reviews_train, cutoff_threshold=14)
    else:
        full_vocab = feat.get_vocab(reviews_train, cutoff_threshold=9) + \
            feat.get_vocab_bigram(reviews_train, cutoff_threshold=14)
    print("\n#features is ", len(full_vocab))

    print("\ngenerating the training matrix ...")
    # training matrix of data and tags
    if feat_type == 'unigram':
        train_matrix = feat.bag_words2vec_unigram(full_vocab, reviews_train)
    elif feat_type == 'bigram':
        train_matrix = feat.bag_words2vec_bigram(full_vocab, reviews_train)
    else:
        train_matrix = feat.concatenate_feat(feat.bag_words2vec_unigram(
            full_vocab, reviews_train), feat.bag_words2vec_bigram(full_vocab, reviews_train))
    if test:
        print('\ndescription of training matrix', stats.describe(train_matrix))

    # training vectors of labels
    train_class_vector = np.hstack(
        (feat.get_class_vec('neg', train_size), feat.get_class_vec('pos', train_size)))

    print("\ntraining the Naive Bayes classifier ...")
    # train the Naive Bayes classifier
    prob_neg_vec, prob_pos_vec, prob_sentiment = train_nb_classifier(
        train_matrix, train_class_vector, smoothing)
    print("prob vector on neg reviews", prob_neg_vec, "\nprob vector on pos reviews",
          prob_pos_vec, "\nprob of sentiment", prob_sentiment)

    # function to see the words with 0 probability (without smoothing)
    def print_feature_zeroval():
        for i in range(len(prob_pos_vec)):
            if prob_pos_vec[i] == 0.0:
                print(full_vocab[i])

    # parameters for testing
    i, neg_correct, pos_correct = 0, 0, 0
    classification_result = [0]*len(reviews_test)  # 0 for misclassification
    print("\ntesting the Naive Bayes classifier ...")
    # test the classifier with another review
    for i in range(len(reviews_test)):
        if feat_type == 'unigram':
            test_vec = feat.words2vec_unigram(full_vocab, reviews_test[i])
        elif feat_type == 'bigram':
            test_vec = feat.words2vec_bigram(full_vocab, reviews_test[i])
        else:
            test_vec = np.array(feat.words2vec_unigram(
                full_vocab, reviews_test[i])) + np.array(feat.words2vec_bigram(full_vocab, reviews_test[i]))

        test_result = test_nb_classifier(
            test_vec, prob_neg_vec, prob_pos_vec, prob_sentiment)
        # neg review result=0
        if i < test_size:
            # print("Test sample %d \nThis review is %d review, 0 for neg" % (i, test_result))
            if test_result == 0:
                neg_correct += 1
                classification_result[i] = 1
        # pos review result=1
        else:
            # print("Test sample %d \nThis review is %d review, 1 for pos" % (i, test_result))
            if test_result == 1:
                pos_correct += 1
                classification_result[i] = 1

    # print overall accuracy
    print("accuracy for negative reviews", (neg_correct/test_size))
    print("accuracy for positive reviews", (pos_correct/test_size))
    print("overall accuracy for this classifier", sum(
        classification_result)/len(classification_result))
    # assert sum(classification_result) / \
    #     len(classification_result) == (neg_correct+pos_correct)/(test_size*2)

    # write results to text file
    f = open('./results.txt', 'a+', encoding='utf-8')
    f.write("\nfeature: %s\t#feature: %d\ttraining size: %d\tsmooth: %s\tneg_accuracy: %f\tpos_accuracy: %f" % (
        feat_type, len(full_vocab), train_size, smoothing, (neg_correct/test_size), (pos_correct/test_size)))
    f.close()
    print("\written to files ...")

    return classification_result


def nb_classifier(feat_type, smoothing, test=False):
    # para smoothing: the type of smoothing
    # para test: test case or not

    print("\npreparing data ...")
    train_size, test_size, reviews_train, reviews_test = prepare_data(test)
    classification_result = nb_classifier_train_test(train_size, test_size, reviews_train, reviews_test, feat_type, smoothing, test)
    return classification_result 
    

def ten_fold_crossvalidation(fold_type, feat_type, test=False):
    # read data (avoid replication)
    neg_reviews = text.read_data_tag_from_file('neg')
    pos_reviews = text.read_data_tag_from_file('pos')

    no_fold, length_data = 10, 1000
    if fold_type == 'consecutive':
        test_ranges = n_fold_cons(no_fold, length_data)
    else:
        test_ranges = n_fold_RR(no_fold, length_data)

    results = list()
    for i in range(len(test_ranges)-7):
        if fold_type == 'consecutive':
            train_size, test_size, reviews_train, reviews_test = prepare_data_tenfold(neg_reviews, pos_reviews, test_ranges[i])
        else:
            train_size, test_size, reviews_train, reviews_test = prepare_data_roundrobin(neg_reviews, pos_reviews, test_ranges[i])
        result = nb_classifier_train_test(train_size, test_size, reviews_train, reviews_test, feat_type, 'laplace', test)
        results.append(result)
    
    performances = np.array(([sum(x)/len(x) for x in results])) # list of accuracies
    perf_average, variance = np.average(performances), np.var(performances)

    # write results to text file
    f = open('./results_final.txt', 'a+', encoding='utf-8')
    f.write("\nfold type: %s\tfeature: %s\tperformances: %s\taverage performance: %f\tvariance: %f" % (fold_type, feat_type, performances, perf_average, variance))
    f.close()
    print("\written to files ...")

    
    