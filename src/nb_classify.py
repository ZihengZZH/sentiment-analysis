import numpy as np
import math
import datetime
import progressbar
from scipy import stats
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from multiprocessing import cpu_count, Pool

import src.text as text
import src.bow_feat as feat
import src.cv_partition as cv
import src.stats_test as st


# save the training model into a file for later use
def save_to_file(prob_neg_vec, prob_pos_vec, prior_sentiment, vocab):
    # NOTE THAT ONLY THE LATEST MODEL WILL BE KEPT UNLESS INTENDED
    # type prob_neg_vec: numpy array
    # type prob_pos_vec: numpy array
    # type prior_sentiment: float
    # type vocab: list(str)
    np.save("./models/nb_models/prob_neg_vector", prob_neg_vec)
    np.save("./models/nb_models/prob_pos_vector", prob_pos_vec)
    np.save("./models/nb_models/prior_sentiment", prior_sentiment)
    np.save("./models/nb_models/vocabulary", vocab)
    readme_notes = np.array(["This model is trained on ", str(datetime.datetime.now())])
    np.savetxt("./models/nb_models/readme.txt", readme_notes, fmt="%s")
    np.savetxt("./models/nb_models/vocabulary_text.txt", vocab, fmt="%s")


# load the training model from the file 
def load_from_file():
    # type prob_neg_vec: numpy array
    # type prob_pos_vec: numpy array
    # type prior_sentiment: float
    # type vocab: list(str)
    try:
        prob_neg_vec = np.load("./models/nb_models/prob_neg_vector.npy")
        prob_pos_vec = np.load("./models/nb_models/prob_pos_vector.npy")
        prior_sentiment = np.load("./models/nb_models/prior_sentiment.npy")
        vocab = np.load("./models/nb_models/vocabulary.npy")
        return prob_neg_vec, prob_pos_vec, prior_sentiment, vocab
    
    except:
        print("\n--THE NB CLASSIFIER HAS NOT BEEN TRAINED --\n")


# train the Naive Bayes classifier & save models to file
def train_nb_classifier(train_mat, train_c, vocab, smooth_type):
    # para train_mat: matrix of data & tags
    # para train_c: R^2 sentiment (neg/0 or pos/1)
    # return para: prob_vec: R^2 vec of conditional prob
    # return para: prior_sentiment
    if (smooth_type != 'laplace') and (smooth_type != 'None'):
        return

    no_train_review = len(train_mat)
    no_feature = len(train_mat[0])
    prior_sentiment = sum(train_c) / float(no_train_review)  # prob 0.5

    # numerator Tct
    prob_neg_num = np.zeros(no_feature)
    prob_pos_num = np.zeros(no_feature)
    # denominator sum(Tct)
    prob_neg_denom = .0
    prob_pos_denom = .0

    # check whether smoothing or not
    if smooth_type == 'laplace':
        k = 1.0  # constant for all words
    else:
        k = 0.0

    bar = progressbar.ProgressBar()

    # iteration on every training review
    for i in bar(range(no_train_review)):
        if train_c[i] == 0:
            prob_neg_num += train_mat[i]
            prob_neg_denom += (sum(train_mat[i])+k)
        else:
            prob_pos_num += train_mat[i]
            prob_pos_denom += (sum(train_mat[i])+k)
    # prob vector for negative reviews P(fi|0)
    prob_neg_vec = np.log((prob_neg_num+k)/(prob_neg_denom))
    # prob vector for positive reviews P(fi|1)
    prob_pos_vec = np.log((prob_pos_num+k)/(prob_pos_denom))

    save_to_file(prob_neg_vec, prob_pos_vec, prior_sentiment, vocab)


# test the Naive Bayes classifier & load models from file
def test_nb_classifier(test_vec):
    # para prob_neg_vec: P(fi|0)
    # para prob_pos_vec: P(fi|1)
    # para prob_class: P(c=0) or P(c=1) (equal in this project)
    prob_neg_vec, prob_pos_vec, prior_class, vocab = load_from_file()

    prob_neg = sum(test_vec*prob_neg_vec) + np.log(1.0-prior_class)
    prob_pos = sum(test_vec*prob_pos_vec) + np.log(prior_class)
    
    # binary classification argmax
    return np.argmax([prob_neg, prob_pos])


# write results to text file
def save_results(feat_type, vocab_length, train_size, smoothing, neg_accuracy, pos_accuracy, classification):
    notes = "results obtained on " + str(datetime.datetime.now())
    f = open('./results/results.txt', 'a+', encoding='utf-8')
    f.write("\nfeature: %s\t#feature: %d\ttraining size: %d\tsmooth: %s\tneg_accuracy: %f\tpos_accuracy: %f\tclassification_accuracy: %f\tnotes: %s" % (
        feat_type, vocab_length, train_size, smoothing, neg_accuracy, pos_accuracy, classification, notes))
    f.close()


# write results to text file
def save_results_cv(fold_type, feat_type, performances, perf_average, variance):
    notes = "results obtained on " + str(datetime.datetime.now())
    f = open('./results/results_cv.txt', 'a+', encoding='utf-8')
    f.write("\nfold type: %s\nfeature: %s\t#performance: %s\taverage performance: %f\tvariance: %f\tnotes: %s" % (
        fold_type, feat_type, performances, perf_average, variance, notes))
    f.close()


# sklearn implementation of Naive Bayes classifier
def sklearn_nb_classifier(train_reviews, train_labels, test_reviews_neg, test_reviews_pos):
    # para train_reviews: training matrix that each row is a sample
    # para train_labels: training labels that each element is 0/1
    # para test_reviews_neg: negative test reviews
    # para test_reviews_pos: positive test reviews
    model = MultinomialNB()
    
    print("\ntraining the sklearn NB classifier ...")
    model.fit(train_reviews, train_labels)

    print("\ntesting the sklearn NB classifier ...")
    predicted_neg = model.predict(test_reviews_neg)
    predicted_pos = model.predict(test_reviews_pos)

    neg_accuracy = (len(test_reviews_neg) - sum(predicted_neg)) / len(test_reviews_neg)
    pos_accuracy = sum(predicted_pos) / len(test_reviews_pos)

    print("\naccuracy for negative reviews", neg_accuracy)
    print("accuracy for positive reviews", pos_accuracy)
    print("overall accuracy for this classifier", (neg_accuracy+pos_accuracy)/2)


# visualise the most important words in the NB classifier
def visual_important_word(which_vec='neg'):
    # para which_vec: visualise which vector (either neg or pos)
    prob_neg_vec, prob_pos_vec, prior_sentiment, vocab = load_from_file()
    prob_vec_visual = prob_neg_vec if which_vec == 'neg' else prob_pos_vec
    word_visual = 'negative' if which_vec == 'neg' else 'positive'
    # perform an indirect parition along the given axis and returns an array indices
    ind = np.argpartition(prob_vec_visual, -10)[-10:]
    
    print("\nvisualise the most important %s words in the NB classifier ..." % word_visual)
    print("---"*20)
    for i in range(len(ind)):
        print(vocab[ind[i]], "\t", prob_vec_visual[ind[i]])
    print("---"*20)


'''  Naive Bayes Classifier  '''
def naive_bayes_classifier(feature_type, smoothing, cv_part=False, train_size_cv=None, test_size_cv=None, reviews_train_cv=None, reviews_test_cv=None):
    # para feature_type: unigram or bigram (n-gram bag of words)
    # para smoothing: the type of smoothing
    # para cv_part: whether or not is is partitioned with cross-validation
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
    train_nb_classifier(train_matrix, train_class_vector, full_vocab, smoothing)
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

        test_result = test_nb_classifier(test_vec)
        if i < test_size:
            '''neg review result=0'''
            # print("Test sample %d \nThis review is %d review, 0 for neg" % (i, test_result))
            if test_result == 0:
                neg_correct += 1
                classification_result[i] = 1
        else:
            '''pos review result=1'''
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
    save_results(feature_type, vocab_length, train_size, smoothing, neg_accuracy, pos_accuracy, overall_accuracy)
    print("\nclassification results written to file")

    return classification_result