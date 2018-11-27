import numpy as np
import src.text as text
import src.bow_feat as feat
import src.nb_classify as nb

if __name__ == "__main__":

    print("\nreading reviews from files ...")
    # read all neg and pos reviews
    neg_reviews = text.read_data_tag_from_file('neg')
    pos_reviews = text.read_data_tag_from_file('pos')
    # train/test partition
    neg_reviews_train = neg_reviews[:900]
    pos_reviews_train = pos_reviews[:900]
    neg_reviews_test = neg_reviews[900:]
    pos_reviews_test = pos_reviews[900:]
    # Note the order: neg, pos
    reviews_train = neg_reviews_train + pos_reviews_train

    print("\nfinding the corpus for the classifier ...")
    # full corpus for the training reviews
    full_corpus = feat.corpus_list(reviews_train)

    print("\ngenerating the training matrix ...")
    # training matrix of data and tags
    train_matrix = feat.bag_words2vec_unigram(full_corpus, reviews_train)

    # training vectors of labels
    train_class_vector = np.hstack((feat.get_class_vec('neg',len(neg_reviews_train)),feat.get_class_vec('pos',len(neg_reviews_train))))
    
    print("\ntraining the Naive Bayes classifier ...")
    # train the Naive Bayes classifier
    prob_neg_vec, prob_pos_vec, prob_sentiment = nb.train_nb_classifier(train_matrix, train_class_vector)
    print("prob vector on neg reviews", prob_neg_vec, "\nprob vector on pos reviews", prob_pos_vec, "\nprob of sentiment", prob_sentiment)
    
    print("\ntesting the Naive Bayes classifier ...")
    # test the classifier with another review
    i, neg_score, pos_score = 1, 0, 0
    for review_test in neg_reviews_test:
        test_vec = feat.words2vec_unigram(full_corpus, review_test)
        test_result = nb.test_nb_classifier(test_vec, prob_neg_vec, prob_pos_vec, prob_sentiment)
        neg_score += test_result
        i += 1
        print("Test sample %d \nThis review is %d review, 0 for neg" % i, test_result)
    for review_test in pos_reviews_test:
        test_vec = feat.words2vec_unigram(full_corpus, review_test)
        test_result = nb.test_nb_classifier(test_vec, prob_neg_vec, prob_pos_vec, prob_sentiment)
        pos_score += test_result
        print("Test sample %d \nThis review is %d review, 1 for pos" % i, test_result)
        i += 1
    
    # print overall accuracy
    print("overall accuracy for negative reviews", 1-(neg_score/1000))
    print("overall accuracy for positive reviews", (pos_score/1000))