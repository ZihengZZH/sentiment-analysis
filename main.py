import numpy as np
import src.text as text
import src.bow_feat as feat
import src.nb_classify as nb
import src.cv_partition as cv
import src.sign_test as st

from scipy import stats
import progressbar
from multiprocessing import cpu_count, Pool

POS_TAGGING = False

def naive_bayes_classifier(feature_type, smoothing, cv_part=False, train_size_cv=None, test_size_cv=None, reviews_train_cv=None, reviews_test_cv=None):
    # para smoothing: the type of smoothing
    # para test: test case or not

    print("\nNaive Bayes Classifier on sentiment detection running\n\npreparing data ...")
    if not cv_part:
        train_size, test_size, reviews_train, reviews_test = cv.prepare_data()
    else:
        train_size, test_size, reviews_train, reviews_test = train_size_cv, test_size_cv, reviews_train_cv, reviews_test_cv
    
    print("\nfinding the corpus for the classifier ...")
    # full vocabulary for the training reviews (frequency cutoff implemented)
    if feature_type == 'unigram':
        full_vocab = feat.get_vocab(reviews_train, cutoff_threshold=9)
    elif feature_type == 'bigram':
        full_vocab = feat.get_vocab_bigram(reviews_train, cutoff_threshold=14)
    else:
        full_vocab = feat.get_vocab(reviews_train, cutoff_threshold=9) + feat.get_vocab_bigram(reviews_train, cutoff_threshold=14)
    vocab_length = len(full_vocab)
    print("\n#features is ", vocab_length)
    
    print("\ngenerating the training matrix ...")
    # training matrix of data
    if feature_type == 'unigram':
        train_matrix = feat.bag_words2vec_unigram(full_vocab, reviews_train)
    elif feature_type == 'bigram':
        train_matrix = feat.bag_words2vec_bigram(full_vocab, reviews_train)
    else:
        train_matrix = feat.concatenate_feat(feat.bag_words2vec_unigram(
            full_vocab, reviews_train), feat.bag_words2vec_bigram(full_vocab, reviews_train))
    print('\ndescription of training matrix', stats.describe(train_matrix))
    
    # training vector of labels
    train_class_vector = np.hstack(
        (feat.get_class_vec('neg', train_size), feat.get_class_vec('pos', train_size)))

    print("\ntraining the Naive Bayes classifier ...")
    # train the Naive Bayes classifier
    nb.train_nb_classifier(
        train_matrix, train_class_vector, smoothing)
    print("\nthe training process, DONE. ")
    
    # parameters for testing
    i, neg_correct, pos_correct = 0, 0, 0
    classification_result = [0]*len(reviews_test)  # 0 for misclassification
    print("\ntesting the Naive Bayes classifier ...")
    # test the classifier with another review
    for i in range(len(reviews_test)):
        if feature_type == 'unigram':
            test_vec = feat.words2vec_unigram(full_vocab, reviews_test[i])
        elif feature_type == 'bigram':
            test_vec = feat.words2vec_bigram(full_vocab, reviews_test[i])
        else:
            test_vec = np.array(feat.words2vec_unigram(
                full_vocab, reviews_test[i])) + np.array(feat.words2vec_bigram(full_vocab, reviews_test[i]))

        test_result = nb.test_nb_classifier(test_vec)
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
    neg_accuracy = (neg_correct/test_size)
    pos_accuracy = (pos_correct/test_size)
    overall_accuracy = sum(classification_result)/len(classification_result)
    print("\naccuracy for negative reviews", neg_accuracy)
    print("accuracy for positive reviews", pos_accuracy)
    print("overall accuracy for this classifier", overall_accuracy)
    
    # save classification results to files
    nb.save_results(feature_type, vocab_length, train_size, smoothing, neg_accuracy, pos_accuracy, overall_accuracy)
    print("\nclassification results written to file")

    return classification_result


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
    naive_bayes_classifier("bigram", "laplace")
    # cross_validation_10fold('consecutive', 'unigram')
