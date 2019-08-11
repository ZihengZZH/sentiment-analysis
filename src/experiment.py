import numpy as np
import progressbar
from scipy import stats
from smart_open import smart_open
from multiprocessing import cpu_count, Pool

import src.text as text
import src.bow_feat as feat
import src.cv_partition as cv
import src.stats_test as st
import src.nb_classify as nb
import src.svm_classify as svm
import src.doc2vec as doc2vec
import src.utility as utility

POS_TAGGING = False


def ten_fold_NB(fold_type, feature_type):
    # read data (avoid replication)
    neg_reviews = text.read_data_from_file('neg')
    pos_reviews = text.read_data_from_file('pos')

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
        result = nb.naive_bayes_classifier(feature_type, 'laplace', True, train_size, test_size, reviews_train, reviews_test)
        results.append(result)

    performances = np.array(([sum(x)/len(x) for x in results]))  # list of accuracies
    perf_average, variance = np.average(performances), np.var(performances)

    # save results into file 
    nb.save_results_cv(fold_type, feature_type, results, performances, perf_average, variance)
    print("\ncross validation results written to file")


def ten_fold_SVM(fold_type, feature_type, if_doc2vec, model_no):
    # read data (avoid replication)
    neg_reviews = text.read_data_from_file('neg')
    pos_reviews = text.read_data_from_file('pos')

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
        result = svm.SVM_classifier(feature_type, True, if_doc2vec, model_no, True, train_size, test_size, reviews_train, reviews_test)
        results.append(result)

    performances = results  # list of accuracies
    perf_average, variance = np.average(performances), np.var(performances)

    # save results into file 
    svm.save_results_cv(fold_type, feature_type, if_doc2vec, results, performances, perf_average, variance)
    print("\ncross validation results written to file")


# Grid search for SVM classifier using bow features
def grid_search_svm_bow():
    svm.SVM_grid_search_bow('unigram')
    svm.SVM_grid_search_bow('bigram')
    svm.SVM_grid_search_bow('both')


# Grid search for SVM classifier using doc2vec embeddings
def grid_search_svm_doc2vec(start_no, concatenate):
    # para start_no:
    # para concatenate:
    if concatenate:
        svm.SVM_grid_search_doc2vec([15, 18], True)
        svm.SVM_grid_search_doc2vec([15, 20], True)
    else:
        for i in range(start_no, 21):
            print("\nmodel no: %d" % i)
            svm.SVM_grid_search_doc2vec(i, False)


def train_doc2vec_models():
    doc2vec.train_doc_embedding()


def examine_doc2vec_results(model_list):
    train_size, test_size, reviews_train, reviews_test = cv.prepare_data()
    for model_no in model_list:
        doc2vec.examine_results(model_no, reviews_train, train_size)


def main():
    ten_fold_NB('consecutive', 'unigram')
    ten_fold_NB('consecutive', 'bigram')
    ten_fold_NB('consecutive', 'both')

    ten_fold_SVM('consecutive', 'unigram', False, 0)
    ten_fold_SVM('consecutive', 'bigram', False, 0)
    ten_fold_SVM('consecutive', 'both', False, 0)

    ten_fold_SVM('consecutive', '#', True, 15)
    ten_fold_SVM('consecutive', '#', True, 18)
    ten_fold_SVM('consecutive', '#', True, 20)

    ten_fold_SVM('consecutive', '#', True, [15, 18])
    ten_fold_SVM('consecutive', '#', True, [15, 20])


def permutation_test():
    results_nb_uni = nb.naive_bayes_classifier('unigram', 'laplace')
    results_nb_bi = nb.naive_bayes_classifier('bigram', 'laplace')
    results_nb_both = nb.naive_bayes_classifier('both', 'laplace')
    result_svm_uni = svm.SVM_classifier('unigram')
    result_svm_bi = svm.SVM_classifier('bigram')
    result_svm_both = svm.SVM_classifier('both')
    result_doc2vec_15 = svm.SVM_classifier('', if_doc2vec=True, model_no=15)
    result_doc2vec_18 = svm.SVM_classifier('', if_doc2vec=True, model_no=18)
    result_doc2vec_15_18 = svm.SVM_classifier('', if_doc2vec=True, model_no=[15,18], concatenate=True)


def visualization_preparation():
    train_size, test_size, reviews_train, reviews_test = cv.prepare_data()
    doc2vec.process_metadata(15, reviews_train, reviews_test)
    doc2vec.process_metadata(18, reviews_train, reviews_test)


if __name__ == "__main__":
    # initialize a timer
    timer = utility.Timer()
    timer.start()
    # function to run
    ten_fold_NB('consecutive', 'unigram')
    timer.stop()