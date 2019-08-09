import numpy as np
from data import *
import progressbar
import math
import random


def n_fold_cons(no_fold, length_data):
    """Consecutive splitting 
    (in case data cannot be separated evenly)
    """
    length_split = int(length_data / no_fold)
    test_range = list()
    for i in range(no_fold):
        test_range.append([length_split*i, length_split*(i+1)])
    return test_range


def n_fold_RR(no_fold, length_data):
    """Round-robin splitting mod 10
    """
    mod = no_fold  # basically the same
    test_splits = list()
    length_split = int(length_data / no_fold)
    for i in range(no_fold):
        test_split = list()
        for j in range(length_split):
            test_split.append(i+mod*j)
        test_splits.append(test_split)
    return test_splits


def partition(neg_reviews, pos_reviews, test=False):
    """train/test partition / cross-validation
    --
    return para: partition size of each class
    """
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


def prepare_data(tags=False, gridsearch=False):
    """prepare the data without partition
    (read all neg and pos reviews)
    """
    if not tags:
        neg_reviews = read_data_from_file('neg')
        pos_reviews = read_data_from_file('pos')
    else:
        neg_reviews = read_data_tag_from_file('neg')
        pos_reviews = read_data_tag_from_file('pos')

    if not gridsearch:
        print("\ntrain/test partitioning ...")
        return partition(neg_reviews, pos_reviews)
    else:
        splits_tenfold = n_fold_cons(10, 1000)
        search_range = splits_tenfold[math.floor(random.random()*10)]
        return prepare_data_tenfold(neg_reviews, pos_reviews, search_range)


def prepare_data_tenfold(neg_reviews, pos_reviews, test_range):
    """prepare the data with consecutive splitting
    --
    para test_range: list[start:end]
    return para: sizes and reviews of each partition
    """
    train_size, test_size = 900, 100
    [start_point, end_point] = test_range
    neg_reviews_train = neg_reviews[:start_point] + neg_reviews[end_point:]
    neg_reviews_test = neg_reviews[start_point:end_point]
    pos_reviews_train = pos_reviews[:start_point] + pos_reviews[end_point:]
    pos_reviews_test = pos_reviews[start_point:end_point]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train   # dimension: 1
    reviews_test = neg_reviews_test + pos_reviews_test      # dimension: 1
    return train_size, test_size, reviews_train, reviews_test


def prepare_data_roundrobin(neg_reviews, pos_reviews, test_range):
    """prepare the data with RR splitting
    --
    para test_range: list[index]
    return para: sizes and reviews of each partition
    """
    train_size, test_size = 900, 100
    neg_reviews_train, neg_reviews_test, pos_reviews_train, pos_reviews_test = [], [], [], []
    for ele in test_range:
        neg_reviews_test += neg_reviews[ele]
        pos_reviews_test += pos_reviews[ele]

    train_range = list(set(range(1000)) - set(test_range))
    for ele in train_range:
        neg_reviews_train += neg_reviews[ele]
        pos_reviews_train += pos_reviews[ele]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train   # dimension: 1
    reviews_test = neg_reviews_test + pos_reviews_test      # dimension: 1
    return train_size, test_size, reviews_train, reviews_test