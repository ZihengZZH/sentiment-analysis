import numpy as np
import progressbar
from scipy import stats
from multiprocessing import cpu_count, Pool

import src.text as text
import src.bow_feat as feat
import src.cv_partition as cv
import src.sign_test as st
import src.nb_classify as nb
import src.svm_classify as svm
import src.doc2vec as doc2vec

POS_TAGGING = False


def cross_validation_10fold(fold_type, feature_type):
    # read data (avoid replication)
    neg_reviews = text.read_data_tag_from_file('neg')
    pos_reviews = text.read_data_tag_from_file('pos')

    no_fold, length_data = 10, 1000
    if fold_type == 'consecutive':
        test_ranges = cv.n_fold_cons(no_fold, length_data)
    else:
        test_ranges = cv.n_fold_RR(no_fold, length_data)

    results = list()
    for i in range(len(test_ranges)):
        if fold_type == 'consecutive':
            train_size, test_size, reviews_train, reviews_test = cv.prepare_data_tenfold(
                neg_reviews, pos_reviews, test_ranges[i])
        else:
            train_size, test_size, reviews_train, reviews_test = cv.prepare_data_roundrobin(
                neg_reviews, pos_reviews, test_ranges[i])
        result = naive_bayes_classifier(feature_type, 'laplace', True, train_size, test_size, reviews_train, reviews_test)
        results.append(result)

    performances = np.array(([sum(x)/len(x) for x in results]))  # list of accuracies
    perf_average, variance = np.average(performances), np.var(performances)

    # save results into file 
    nb.save_results_cv(fold_type, feature_type, performances, perf_average, variance)
    print("\ncross validation results written to file")


if __name__ == "__main__":
    # nb.naive_bayes_classifier("unigram", "laplace")
    # cross_validation_10fold('consecutive', 'unigram')
    