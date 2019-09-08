import os
import json
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.utils.display import print_new


class RFClassifier(object):
    """
    Random Forest Classifier
    --
    # para embedding: bow / word2vec / doc2vec / lstm
    # para test: whether or not to test
    """
    def __init__(self, embedding, test=False):
        print_new("Random Forest Classifier on sentiment analysis")
        self.name = 'rf_%s' % embedding
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.hparams = dict()
        self.accuracy = None
        self.recall = None
        self.f1_score = None
        self.config = json.load(open('./config.json', 'r'))
        self.save_path = os.path.join(self.config['data_path'],
                                    self.config['rf_classifier']['save_path'], 
                                    self.name)
        self.config = self.config['rf_classifier']
        self.hparams_file = os.path.join(self.config['hparams_path'],
                                        '%s.json' % self.name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
    def save_model(self):
        json_file = open(self.hparams_file, 'w')
        json.dump(self.hparams, json_file, indent=4)
        print_new("HPARAMS SAVED")
    
    def load_model(self):
        if not os.path.isfile(self.hparams_file):
            return False
        self.hparams = json.load(open(self.hparams_file, 'r'))
        print_new("HPARAMS LOADED")
        return True
    
    def load_data(self, train_mat, train_label, test_mat, test_label):
        assert isinstance(train_mat, np.ndarray), "train data format error"
        assert isinstance(test_mat, np.ndarray), "test data format error"
        self.X_train = train_mat
        self.y_train = train_label
        self.X_test = test_mat
        self.y_test = test_label
        print_new("LOADING EMBEDDINGS COMPLETED")
    
    def train_model(self):
        if not self.X_train.any() or \
            not self.y_train.any() or \
            not self.X_test.any() or \
            not self.y_test.any():
            print_new("TRAINING CANNOT PROCEED")
        if not self.load_model():
            self.finetune_model()
        
        self.model = RandomForestClassifier(
                    n_estimators=self.hparams['n_estimators'],
                    max_features=self.hparams['max_features'],
                    max_depth=self.hparams['max_depth'],
                    criterion=self.hparams['criterion'],
                    verbose=1, n_jobs=-1)
        
        print_new("training sklearn RF classifier")
        self.model.fit(self.X_train, self.y_train)
        print_new("training sklearn RF classifier completed")
    
    def finetune_model(self):
        clf = GridSearchCV(RandomForestClassifier(), 
                            self.config['hparams_set'], 
                            cv=5, n_jobs=-1, verbose=1)
        clf.fit(self.X_train, self.y_train)
        print_new(clf.best_params_)
        print_new(clf.best_score_)
        self.hparams = clf.best_params_
        self.save_model()
    
    def evaluate_model(self):
        if not self.X_train.any() or \
            not self.y_train.any() or \
            not self.X_test.any() or \
            not self.y_test.any():
            print_new("EVALUATION CANNOT PROCEED")
        
        print_new("evaluating sklearn RF classifier")
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred, average='macro')
        self.f1_score = f1_score(self.y_test, y_pred, average='macro')

    def save_classification(self):
        dict2save = dict()
        dict2save['classifier_name'] = 'rf'
        dict2save['embedding'] = self.name
        dict2save['accuracy'] = self.accuracy
        dict2save['recall'] = self.recall
        dict2save['f1-score'] = self.f1_score
        dict2save['time'] = datetime.datetime.now()