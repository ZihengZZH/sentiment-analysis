import numpy as np
import src.text as text
import src.bow_feat as feat
import src.nb_classify as nb
import src.sign_test as st


# implement Naive Bayes and report results using simple classification accuracy
def part_1():
    # nb_classifier(feature_type, smoothing)

    nb.nb_classifier('unigram', 'None') # done
    nb.nb_classifier('bigram', 'None') # done
    nb.nb_classifier('both', 'None')

    nb.nb_classifier('unigram', 'laplace') # done
    nb.nb_classifier('bigram', 'laplace') # done
    nb.nb_classifier('both', 'laplace')


def part_2():
    # unigram feature
    # unigram_result_no_smoothing = nb.nb_classifier('unigram', 'None')
    # unigram_result_smoothing = nb.nb_classifier('unigram', 'laplace')
    # st.run_sign_test(unigram_result_no_smoothing, unigram_result_smoothing, 'unigram')

    # bigram feature
    bigram_result_no_smoothing = nb.nb_classifier('bigram', 'None')
    bigram_result_smoothing = nb.nb_classifier('bigram', 'laplace')
    st.run_sign_test(bigram_result_no_smoothing, bigram_result_smoothing, 'bigram')

    # both unigram and bigram features
    # both_result_no_smoothing = nb.nb_classifier('both', 'None')
    # both_result_smoothing = nb.nb_classifier('both', 'laplace')
    # st.run_sign_test(both_result_no_smoothing, both_result_smoothing, 'both') # darwin mac


def part_3():
    nb.ten_fold_crossvalidation('consecutive', 'unigram') 
    nb.ten_fold_crossvalidation('consecutive', 'bigram') # T-580

def part_4():
    nb.ten_fold_crossvalidation('RR', 'unigram')
    nb.ten_fold_crossvalidation('RR', 'bigram') # Macbook-pro


if __name__ == "__main__":
    # part_1()
    # part_2()
    # part_3()
    # part_4()
    nb.ten_fold_crossvalidation('RR', 'unigram') 
    