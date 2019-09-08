import os
import json
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.utils.display import print_new


class NBClassifier(object):
    """
    Naive Bayes Classifier
    --
    # para embedding: bow / word2vec / doc2vec / lstm
    # para smoothing: laplace / None
    # para test: whether or not to test
    """
    def __init__(self, embedding, smoothing, test=False):
        print_new("Naive Bayes Classifier on sentiment analysis")
        self.name = 'nb_%s' % embedding
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        if smoothing == 'None' or smoothing == 'laplace':
            self.smoothing = smoothing
        self.model = None
        self.vocabulary = None
        self.prob_neg_vec = None
        self.prob_pos_vec = None
        self.prior_sentiment = None
        self.accuracy = None
        self.recall = None
        self.f1_score = None
        self.config = json.load(open('./config.json', 'r'))
        self.save_path = os.path.join(self.config['data_path'],
                                    self.config['nb_classifier']['save_path'], 
                                    self.name)
        self.config = self.config['nb_classifier']
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
    
    def save_model(self):
        np.save(os.path.join(self.save_path, "prob_neg_vector"), self.prob_neg_vec)
        np.save(os.path.join(self.save_path, "prob_pos_vector"), self.prob_pos_vec)
        np.save(os.path.join(self.save_path, "prior_sentiment"), self.prior_sentiment)
        np.save(os.path.join(self.save_path, "vocabulary"), self.vocabulary)
        print_new("SAVING NB CLASSIFIER COMPLETED")

    def load_model(self):
        try:
            self.prob_neg_vec = np.load(os.path.join(self.save_path, "prob_neg_vector.npy"))
            self.prob_pos_vec = np.load(os.path.join(self.save_path, "prob_pos_vector.npy"))
            self.prior_sentiment = np.load(os.path.join(self.save_path, "prior_sentiment.npy"))
            self.vocabulary = np.load(os.path.join(self.save_path, "vocabulary.npy"))
            print_new("LOADING NB CLASSIFIER COMPLETED")
        except:
            print_new("NB CLASSIFIER HAS NOT BEEN TRAINED\n")
    
    def load_data(self, train_mat, train_label, test_mat, test_label, vocab):
        """load representations / embeddings
        --
        # type train_mat: np.array()
        # type test_mat: np.array()
        # type vocab: dict()
        """
        assert isinstance(train_mat, np.ndarray), "train data format error"
        assert isinstance(test_mat, np.ndarray), "test data format error"
        self.X_train = train_mat
        self.y_train = train_label
        self.X_test = test_mat
        self.y_test = test_label
        self.vocabulary = vocab
        print_new("LOADING EMBEDDINGS COMPLETED")
    
    def train_model(self):
        """train self-implemented NB classifier
        """
        if not self.X_train.any() or \
            not self.y_train.any() or \
            not self.X_test.any() or \
            not self.y_test.any():
            print_new("TRAINING CANNOT PROCEED")
        
        print_new("training self-implemented NB classifier")
        train_length = len(self.X_train)
        embedding_length = len(self.X_train[0])
        self.prior_sentiment = sum(self.y_train) / train_length

        # numberator Tct
        prob_neg_num = np.zeros(embedding_length)
        prob_pos_num = np.zeros(embedding_length)
        # denominator sum(Tct)
        prob_neg_denom = .0
        prob_pos_denom = .0
        
        k = 2.0 if self.smoothing == 'laplace' else 0.0
        for i in tqdm(range(train_length)):
            if self.y_train == 0:
                prob_neg_num += self.X_train[i]
                prob_neg_denom += sum(self.X_train[i])
            else:
                prob_pos_num += self.X_train[i]
                prob_pos_denom += sum(self.X_train[i])
        
        # prob vector for negatives P(fi|0)
        self.prob_neg_vec = np.log((prob_neg_num+k) / (prob_neg_denom+k))
        # prob vector for positive P(fi|1)
        self.prob_pos_vec = np.log((prob_pos_num+k) / (prob_pos_denom+k))
        print_new("training self-implemented NB classifier completed")

    def evaluate_model(self):
        """evaluate self-implemented NB classifier
        """
        if not self.X_train.any() or \
            not self.y_train.any() or \
            not self.X_test.any() or \
            not self.y_test.any():
            print_new("EVALUATION CANNOT PROCEED")
        
        print_new("evaluating self-implemented NB classifier")
        y_pred = np.zeros(len(self.y_test))
        for i in tqdm(range(len(self.X_test))):
            test_vec = self.X_test[i]
            prob_neg = sum(test_vec*self.prob_neg_vec) + np.log(1.0-self.prior_sentiment)
            prob_pos = sum(test_vec*self.prob_pos_vec) + np.log(self.prior_sentiment)
            # binary classification argmax
            y_pred[i] = np.argmax([prob_neg, prob_pos])
        
        print(classification_report(self.y_test, y_pred))
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred, average='macro')
        self.f1_score = f1_score(self.y_test, y_pred, average='macro')
    
    def train_model_sklearn(self):
        """train sklearn NB classifier
        """
        if not self.X_train or not self.y_train or \
        not self.X_test or not self.y_test:
            print_new("TRAINING CANNOT PROCEED")
        
        print_new("training sklearn NB classifier")
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)
        print_new("training sklearn NB classifier completed")

    def evaluate_model_sklearn(self):
        """evaluate sklearn NB classifier
        """
        if not self.X_train or not self.y_train or \
        not self.X_test or not self.y_test:
            print_new("EVALUATION CANNOT PROCEED")
        
        print_new("evaluating self-implemented NB classifier")
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1_score = f1_score(self.y_test, y_pred)
    
    def visualize_important_word(self):
        """visualize the most important words in NB classifier
        """
        print_new("visualizing the most important words in NEGATIVE")
        # perform an indirect partition along the given axis and returns an array indices
        ind = np.argpartition(self.prob_neg_vec, -10)[-10:]
        for i in range(len(ind)):
            print(self.vocabulary[ind[i]], "\t", self.prob_neg_vec[ind[i]])

        print_new("visualizing the most important words in POSITIVE")
        # perform an indirect partition along the given axis and returns an array indices
        ind = np.argpartition(self.prob_pos_vec, -10)[-10:]
        for j in range(len(ind)):
            print(self.vocabulary[ind[i]], "\t", self.prob_pos_vec[ind[i]])
    
    def save_classification(self):
        """save classification results
        """
        dict2save = dict()
        dict2save['classifier_name'] = 'nb'
        dict2save['embedding'] = self.name
        dict2save['vocabulary'] = len(self.vocabulary)
        dict2save['smoothing'] = self.smoothing
        dict2save['accuracy'] = self.accuracy
        dict2save['recall'] = self.recall
        dict2save['f1-score'] = self.f1_score
        dict2save['time'] = datetime.datetime.now()
