import os
import json
import datetime
import numpy as numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.utils.display import print_new


class RFClassifier(object):
    def __init__(self, embedding):
        print_new("Random Forest Classifier on sentiment analysis")
        self.name = 'rf_%s' % embedding
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.accuracy = None
        self.recall = None
        self.f1_score = None
        self.config = json.load(open('./config.json', 'r'))['rf_classifier']
        self.save_path = os.path.join(self.config['save_path'], self.name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
    def save_model(self):
        pass
    
    def load_model(self):
        pass
    
    def load_data(self, train_mat, train_label, test_mat, test_label):
        assert isinstance(train_mat, np.array()), "train data format error"
        assert isinstance(test_mat, np.array()), "test data format error"
        self.X_train = train_mat
        self.y_train = train_label
        self.X_test = test_mat
        self.y_test = test_label
        print_new("LOADING EMBEDDINGS COMPLETED")
    
    def train_model(self):
        if not self.X_train or not self.y_train or \
        not self.X_test or not self.y_test:
            print_new("TRAINING CANNOT PROCEED")
        if not self.model:
            self.finetune_model()
        
        print_new("training sklearn RF classifier")
        self.model.fit(self.X_train, self.y_train)
        print_new("training sklearn RF classifier completed")
    
    def finetune_model(self):
        clf = GridSearchCV(RandomForestClassifier(), self.config['hparams'], cv=5, n_jobs=-1, verbose=3)
        print_new(clf.best_params_, clf.best_score_)
        self.model = RandomForestClassifier(
                    n_estimators=clf.best_params_['n_estimators'],
                    max_features=clf.best_params_['max_features'],
                    max_depth=clf.best_params_['max_depth'],
                    criterion=clf.best_params_['criterion'],
                    verbose=1, n_jobs=-1)
    
    def evaluate_model(self):
        if not self.X_train or not self.y_train or \
        not self.X_test or not self.y_test:
            print_new("EVALUATION CANNOT PROCEED")
        
        print_new("evaluating sklearn RF classifier")
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1_score = f1_score(self.y_test, y_pred)

    def save_classification(self):
        dict2save = dict()
        dict2save['classifier_name'] = 'rf'
        dict2save['embedding'] = self.name
        dict2save['accuracy'] = self.accuracy
        dict2save['recall'] = self.recall
        dict2save['f1-score'] = self.f1_score
        dict2save['time'] = datetime.datetime.now()