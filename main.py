import numpy as np
import src.text as text
import src.bow_feat as feat
import src.nb_classify as nb

if __name__ == "__main__":

    print("\nreading reviews from files ...")
    # read 10 neg and pos reviews
    neg_reviews = text.read_data_tag_from_file('neg')[:10]
    pos_reviews = text.read_data_tag_from_file('pos')[:10]
    # Note the order: neg, pos
    reviews = neg_reviews + pos_reviews 

    print("\nfinding the corpus for the classifier ...")
    # full corpus for the selected reviews
    full_corpus = feat.corpus_list(reviews)

    print("\ngenerating the training matrix ...")
    # training matrix of data and tags
    train_matrix = feat.bag_words2vec_unigram(full_corpus, reviews)

    # training vectors of labels
    train_class_vector = np.hstack((feat.get_class_vec('neg',len(neg_reviews)),feat.get_class_vec('pos',len(pos_reviews))))
    
    print("\ntraining the Naive Bayes classifier ...")   
    # train the Naive Bayes classifier
    prob_neg_vec, prob_pos_vec, prob_sentiment = nb.train_nb_classifier(train_matrix, train_class_vector)
    print("prob vector on neg reviews", prob_neg_vec, "\nprob vector on pos reviews", prob_pos_vec, "\nprob of sentiment", prob_sentiment)
    
    print("\ntesting the Naive Bayes classifier ...")
    # test the classifier with another review
    test_review = feat.bag_words2vec_unigram(full_corpus, text.read_data_tag_from_file('neg')[11:12])
    test_result = nb.test_nb_classifier(test_review, prob_neg_vec, prob_pos_vec, prob_sentiment)
    print("This review is %d review, 0 for neg and 1 for pos" % test_result)