import os
import gensim
import datetime
import subprocess
import progressbar
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from random import sample
from random import shuffle
from smart_open import smart_open
from nltk.tokenize import word_tokenize
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import src.cv_partition as cv


NO_CORE = multiprocessing.cpu_count() * 3
NO_EPOCH = 10


def tag_reviews(reviews, tag_type):
    # para reviews: movie reviews (either neg or pos)
    # para label_type: either neg or pos
    tagged_document = []
    for i, w in enumerate(reviews):
        tag = '%s_%s' % (tag_type, i)
        tagged_document.append(TaggedDocument(words=w, tags=tag))
    return tagged_document


def save_model(model2save, model_type):
    # para model2save: a Doc2Vec object (dbow or dm)
    # model_type: str indicating dbow or dm
    model2save.save("./models/doc2vec_models/doc2vec.model")
    readme_notes = np.array(["This %s model is trained on %s" % (model_type, str(datetime.datetime.now()))])
    np.savetxt("./models/doc2vec_models/readme.txt", readme_notes, fmt="%s")
    

def save_vectors(train_vecs, test_vecs, model_type):
    # para train_vecs: train data, vectorized by doc2vec model
    # para test_vecs: test data, vectorized by doc2vec model
    # para model_type: str indicating dbow or dm
    np.save("./models/doc2vec_models/train_vectors", train_vecs)
    np.save("./models/doc2vec_models/test_vectors", test_vecs)


def para_embedding(model_type):
    train_size, test_size, reviews_train, reviews_test = cv.prepare_data()
    
    train_data = tag_reviews(reviews_train, 'TRAIN')
    test_data = tag_reviews(reviews_test, 'TEST')
    train_label = np.concatenate((np.ones(train_size)*-1, np.ones(train_size))) 
    test_label = np.concatenate((np.ones(test_size)*-1, np.ones(test_size))) 
    
    '''
    --dm-- defines the training algorithm 
    If dm == 1 means distributed memory (PV-DM),
    and dm == 0 means distributed bag of words (PV-DBOW).
    Distributed memory model preserves the word order in a document whereas Distributed bag of words just uses the bag of words approach, which does not preserve any word order.
    '''

    if not os.path.isfile("./models/doc2vec_models/doc2vec.model"):
        model = None
        if model_type == 'dbow':
            # PV-DBOW plain
            model_dbow = doc2vec.Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, epochs=20, workers=NO_CORE)
            model = model_dbow
            print("\nDBOW doc2vec model initialized.")
        elif model_type == 'dm':
            # PV-DM w/ default averaging
            # a higher starting alpha may improve CBOW/PV-DM models
            model_dm = doc2vec.Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, epochs=20, workers=NO_CORE, alpha=0.05, comment='alpha=0.05')
            model = model_dm
            print("\nDM doc2vec model initialized.")
        else:
            print("Please indicate the type of doc2vec model!")
            return

        print("\nbuild the vocabulary of doc2vec model ...")
        model.build_vocab(train_data + test_data)

        print("\nbegin training the doc2vec model ...")
        bar_train = progressbar.ProgressBar()
        for i in bar_train(range(NO_EPOCH)):
            # perm = np.random.permutation(len(train_data))
            model.train(train_data, total_examples=len(train_data), epochs=model.iter)
        
        bar_test = progressbar.ProgressBar()
        for j in bar_test(range(NO_EPOCH)):
            # perm = np.random.permutation(len(test_data))
            model.train(test_data, total_examples=len(test_data), epochs=model.iter)
        
        save_model(model, model_type)
        print("\ntraining DONE and model saved to file")
    else:
        print("\nmodel existed, loaded from file")
        model = doc2vec.Doc2Vec.load("./models/doc2vec_models/doc2vec.model")

    # get vectors
    def getVecs(model, corpus, size):
        vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    train_vecs = getVecs(model, train_data, 100) # vector size = 100
    test_vecs = getVecs(model, test_data, 100) # vector size = 100
    save_vectors(train_vecs, test_vecs, model_type)

    print("\ntraining & test (vector) data has been written to file ...")




def simple_tryout():
    data = ["I love machine learning. Its awesome.",
            "I love coding in python",
            "I love building chatbots",
            "they chat amagingly well"]

    tagged_data = [doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]


    MAX_EPOCHS = 100 # 10 or 20
    VEC_SIZE = 20
    ALPHA = 0.025

    model = doc2vec.Doc2Vec(size=VEC_SIZE, alpha=ALPHA, min_alpha=0.00025, min_count=1, dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(MAX_EPOCHS):
        print("iteration %d" % epoch)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("./models/doc2vec_models/doc2vec.model")
    print("Model saved")

    model = doc2vec.Doc2Vec.load("./models/doc2vec_models/doc2vec.model")
    # find the vector of a document which is not in training data
    test_data = word_tokenize("I love chatbots".lower())
    v1 = model.infer_vector(test_data)
    print("V1 infer", v1)

    # find most similar doc using tags
    similar_doc = model.docvecs.most_similar('1')
    print(similar_doc)

    # find vector of doc in training data using tags or in other words
    print(model.docvecs['1'])


def imdb_doc2vec():
    # this data object class suffices as a 'TaggedDocument' (with words and tags)
    # plus adds other state helpful for our later evaluation/reporting
    sentiment_document = namedtuple('Sentiment_Document', 'words tags split sentiment')

    alldocs = []
    with smart_open('./dataset/aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no]
            split = ['train', 'test', 'extra', 'extra'][line_no // 25000] # 25k train, 25k test, 25k extra
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//25000]
            alldocs.append(sentiment_document(words, tags, split, sentiment))

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    print("%d docs: %d train-sentiment, %d test-sentiment" % (len(alldocs), len(train_docs), len(test_docs)))

    doc_list = alldocs[:]
    shuffle(doc_list)

    cores = multiprocessing.cpu_count() * 3
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    simple_models = list()
    # PV-DBOW plain
    simple_models.append(doc2vec.Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, epochs=20, workers=cores))
    # PV-DM w/ default averaging, a higher starting alpha may improve CBOW/PV-DM models
    simple_models.append(doc2vec.Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'))

    for model in simple_models:
        model.build_vocab(alldocs)
        print("%s vocabulay scanned & state initialized" % model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])

    def logistic_predictor_from_data(train_targets, train_regressors):
        """Fit a statsmodel logistic predictor on supplied data"""
        logit = sm.Logit(train_targets, train_regressors)
        predictor = logit.fit(disp=0)
        # print(predictor.summary())
        return predictor

    def error_rate_for_model(test_model, train_set, test_set, 
                            reinfer_train=False, reinfer_test=False, 
                            infer_steps=None, infer_alpha=None, infer_subsample=0.2):
        """Report error rate on test_doc sentiments, using supplied model and train_docs"""

        train_targets = [doc.sentiment for doc in train_set]
        if reinfer_train:
            train_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in train_set]
        else:
            train_regressors = [test_model.docvecs[doc.tags[0]] for doc in train_set]
        train_regressors = sm.add_constant(train_regressors)
        predictor = logistic_predictor_from_data(train_targets, train_regressors)

        test_data = test_set
        if reinfer_test:
            if infer_subsample < 1.0:
                test_data = sample(test_data, int(infer_subsample * len(test_data)))
            test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
        else:
            test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
        test_regressors = sm.add_constant(test_regressors)
        
        # Predict & evaluate
        test_predictions = predictor.predict(test_regressors)
        corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
        errors = len(test_predictions) - corrects
        error_rate = float(errors) / len(test_predictions)
        return (error_rate, errors, len(test_predictions), predictor)


    error_rates = defaultdict(lambda: 1.0)

    for model in simple_models:
        print("training %s" % model)
        model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)

        print("\nEvaluating %s" % model)
        err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
        error_rates[str(model)] = err_rate
        print("\n%f %s\n" % (err_rate, model))

    for model in [models_by_name['dbow+dmm']]: 
        print("\nEvaluating %s" % model)
        err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
        error_rates[str(model)] = err_rate
        print("\n%f %s\n" % (err_rate, model))


    print("Err_rate model")
    for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
        print("%f %s" % (rate, name))
