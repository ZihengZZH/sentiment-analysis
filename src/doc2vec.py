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


NO_CORE = multiprocessing.cpu_count() * 3
NO_EPOCH = 20
SIZE_VECTOR = 100
CONTEXT_WINDOW = 10
NEGATIVE = 0
HIERARCHICAL_SOFTMAX = 1

DOC2VEC_PATH = './models/doc2vec_models/'
DOC2VEC_LIST_PATH = './models/doc2vec_models/model_list.txt'


# load the IMDB data to train doc2vec models
def load_IMDB_data():
    sentiment_document = namedtuple('Sentiment_Document', 'words tags split sentiment')
    alldocs = []
    with smart_open('./dataset/aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no]
            split = ['train', 'test', 'extra', 'extra'][line_no // 25000] # 25k train, 25k test, 25k extra as pos, 25k extra as neg
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no // 12500] # [12.5k pos, 12.5k neg] * 2 then unknown
            alldocs.append(sentiment_document(words, tags, split, sentiment))
    return alldocs


# save the doc2vec models
def save_model(model2save, model_name):
    # para model2save: a Doc2Vec object (dbow or dm)
    # para model_name: str
    if os.path.isdir(DOC2VEC_PATH + model_name):
        print("\nmodel %s already existed" % model_name)
        return 
    else:
        save_dir = DOC2VEC_PATH + model_name + '/'
        os.mkdir(save_dir)
        model2save.save(save_dir + 'doc2vec.model')
        readme_notes = np.array(["This %s model is trained on %s" % (model_name, str(datetime.datetime.now()))])
        np.savetxt(save_dir + 'readme.txt', readme_notes, fmt="%s")
    

# load the doc2vec model
def load_model(model_no):
    # para model_no: mark each model with an index
    with smart_open(DOC2VEC_LIST_PATH, 'rb', encoding='utf-8') as model_paths:
        for line_no, line in enumerate(model_paths):
            if line_no == model_no - 1:
                model_path = line.replace('\n', '')
                model = doc2vec.Doc2Vec.load(model_path)
                break
    return model


# save the training / test vectors using doc2vec
def save_vectors(train_targets, train_vecs, test_targets, test_vecs, model_name):
    # para train_targets:
    # para train_vecs: train data, vectorized by doc2vec model
    # para test_targets:
    # para test_vecs: test data, vectorized by doc2vec model
    # para model_name: str
    if os.path.isdir(DOC2VEC_PATH + model_name):
        print("\nmodel %s existed, ready to save vectors" % model_name)
        save_dir = DOC2VEC_PATH + model_name + '/'
        # save vectors 
        np.save(save_dir + 'train_targets', train_targets)
        np.save(save_dir + 'train_vectors', train_vecs)
        np.save(save_dir + 'test_targets', test_targets)
        np.save(save_dir + 'test_vectors', test_vecs)
    else:
        print("\nmodel %s has not been trained" % model_name)


# load the training / test vectors using doc2vec
def load_vectors(model_name):
    # para model_name: str
    if os.path.isdir(DOC2VEC_PATH + model_name):
        save_dir = DOC2VEC_PATH + model_name + '/'
        train_targets = np.load(save_dir + 'train_targets.npy')
        train_vecs = np.load(save_dir + 'train_vectors.npy')
        test_targets = np.load(save_dir + 'test_targets.npy')
        test_vecs = np.load(save_dir + 'test_vectors.npy')
        return train_targets, train_vecs, test_targets, test_vecs
    else:
        print("\nmodel %s has not been trained" % model_name)


# prepare the vectors in the later learning
def vector_4_learning(model, sentiment_reviews):
    # para model: a Doc2Vec object 
    # para sentiment_reviews
    vectors = [model.infer_vector(doc.words, alpha=.1) for doc in sentiment_reviews]
    labels = [doc.sentiment for doc in sentiment_reviews]
    return vectors, labels


# train the doc2vec 
def train_doc_embedding(test=False):
    # para model_type: str indicating dbow or dm
    alldocs = load_IMDB_data()
    print("\nIMDB movie reviews loaded")

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    print("%d docs: %d train-sentiment, %d test-sentiment" % (len(alldocs), len(train_docs), len(test_docs)))

    doc_list = alldocs[:]
    shuffle(doc_list)

    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    '''
    --dm-- defines the training algorithm 
    If dm == 1 means distributed memory (PV-DM),
    and dm == 0 means distributed bag of words (PV-DBOW).
    Distributed memory model preserves the word order in a document whereas Distributed bag of words just uses the bag of words approach, which does not preserve any word order.
    '''

    if test:
        model = doc2vec.Doc2Vec(dm=0, vector_size=SIZE_VECTOR, negative=NEGATIVE, hs=0, min_count=2, sample=0, epochs=NO_EPOCH, workers=NO_CORE)
        print("\nDBOW doc2vec model initialized.")

        print("\nbuild the vocabulary of doc2vec model ...")
        model.build_vocab(alldocs)
        print("%s vocabulay scanned & state initialized" % model)

        model_name = str(model)

        print("\nbegin training the %s doc2vec model ..." % model_name)
        model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)
        save_model(model, model_name)
        print("\ntraining DONE and model saved to file")
    else:
        model_list = []
        
        # PV-DBOW plain
        model_list.append(doc2vec.Doc2Vec(dm=0, vector_size=SIZE_VECTOR, negative=NEGATIVE, hs=HIERARCHICAL_SOFTMAX, min_count=2, sample=0, epochs=NO_EPOCH, workers=NO_CORE))
        print("\nDBOW doc2vec model initialized.")
    
        # PV-DM w/ default averaging
        # a higher starting alpha may improve CBOW/PV-DM models
        model_list.append(doc2vec.Doc2Vec(dm=1, vector_size=SIZE_VECTOR, window=CONTEXT_WINDOW, negative=NEGATIVE, hs=HIERARCHICAL_SOFTMAX, min_count=2, sample=0, epochs=NO_EPOCH, workers=NO_CORE, alpha=0.05, comment='alpha=0.05'))
        print("\nDM doc2vec model initialized.")

        # train and save different doc2vec models
        for model in model_list:
            print("\nbuild the vocabulary of doc2vec model ...")
            model.build_vocab(alldocs)
            print("%s vocabulay scanned & state initialized" % model)

            model_name = str(model).replace('/', '-')
            
            print("\nbegin training the %s doc2vec model ..." % model_name)
            model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)
            save_model(model, model_name)
            print("\ntraining DONE and %s doc2vec model saved to file" % model_name)
        
        print("\nALL doc2vec models saved.")


# infer the doc2vec embedding for train/test reviews
def infer_embedding(model_no, reviews, reviews_size, concatenate):
    # para model_no: which Doc2Vec model / list if concatenate
    # para reviews: all reviews (training or test) 
    # type reviews: list(list(str))
    # type reviews_size: int
    # para concatenate: whether to concatenate two doc2vec models
    if not concatenate:
        model = load_model(model_no)
        print("description of the doc2vec model\t", str(model))
    else:
        model_1 = load_model(model_no[0])
        model_2 = load_model(model_no[1])
        print("description of the 1st doc2vec model\t", str(model_1))
        print("description of the 2nd doc2vec model\t", str(model_2))
        model = ConcatenatedDoc2Vec([model_1, model_2])
    
    sentiment_review = namedtuple('Sentiment_Review', 'words tags sentiment')
    allreviews = []
    bar = progressbar.ProgressBar()
    for i in bar(range(len(reviews))):
        tags = [i]
        if i < reviews_size:
            allreviews.append(sentiment_review(reviews[i], tags, 0.))
        else:
            allreviews.append(sentiment_review(reviews[i], tags, 1.))
    # infer the doc2vec embeddings with given model 
    vectors, labels = vector_4_learning(model, allreviews)
    return vectors, labels


def test_doc_embedding():
    alldocs = load_IMDB_data()
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    doc2vec_model = load_model(2)

    def logistic_predictor_from_data(train_targets, train_regressors):
        """Fit a statsmodel logistic predictor on supplied data"""
        logit = sm.Logit(train_targets, train_regressors)
        predictor = logit.fit(disp=0)
        # print(predictor.summary())
        return predictor
    
    train_targets = [doc.sentiment for doc in train_docs]
    train_regressors = [doc2vec_model.docvecs[doc.tags[0]] for doc in train_docs]
    test_regressors = [doc2vec_model.docvecs[doc.tags[0]] for doc in test_docs]

    predictor = logistic_predictor_from_data(train_targets, train_regressors)
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_docs])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    print("accuracy", corrects/len(test_predictions))
    print(error_rate, errors, len(test_predictions), predictor)


'''CHECK WHETHER DOC2VEC MAKES SENSE'''
def examine_results(model_no, reviews, reviews_size):
    # para model_no: which Doc2Vec to use
    # para reviews: test reviews to examine the results
    alldocs = load_IMDB_data()
    sentiment_review = namedtuple('Sentiment_Review', 'words tags sentiment')
    allreviews = []
    bar = progressbar.ProgressBar()
    for i in bar(range(len(reviews))):
        tags = [i]
        if i < reviews_size:
            allreviews.append(sentiment_review(reviews[i], tags, 0.))
        else:
            allreviews.append(sentiment_review(reviews[i], tags, 1.))
    
    # load the model & generate the target doc id
    model = load_model(model_no)
    doc_id = np.random.randint(0, len(allreviews))

    # save results to file
    with smart_open('./results/examine_doc2vec_%d.txt' % model_no, 'w', encoding='utf-8') as f:
        
        '''ARE INFERRED VECTORS CLOSE TO PRECALCULATED ONES?'''
        print("for doc %d ..." % doc_id)
        f.write("for doc %d ..." % doc_id)
        f.write("\n")
        # infer document vector of the target
        inferred_docvec = model.infer_vector(allreviews[doc_id].words)
        print("%s:\n %s" % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))
        f.write("%s:\n %s" % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))
        f.write("\n")

        ''''DO CLOSE DOCUMENTS SEEM MORE RELATED THAN DISTANT ONES?'''
        sims = model.docvecs.most_similar(doc_id, topn=len(allreviews)) # get all similar documents
        print("Target (%d): <<%s>>\n" % (doc_id, ' '.join(allreviews[doc_id].words)))
        f.write("Target (%d): <<%s>>\n" % (doc_id, ' '.join(allreviews[doc_id].words)))
        f.write("\n")
        # output the most, median, least similar documents to the target
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print("%s %s: <<%s>>\n" % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))
            f.write("%s %s: <<%s>>\n" % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))
            f.write("\n")
        
        '''DO WORD VECTORS SHOW USEFUL SIMILARITIES?'''
        word = 'great'
        # choose the top 20 similar words to 'great'
        similar_words = str(model.wv.most_similar(word, topn=20))
        print("most similar word %s for model %s are as follows\n" % (word, model))
        f.write("most similar word %s for model %s are as follows\n" % (word, model))
        f.write("\n")
        print(similar_words)
        f.write(similar_words)
        f.write("\n")
    
    print("\nexamination, DONE.")


'''PREPARE DATA FOR TENSORFLOW VISUALIZATION'''
def process_metadata(model_no, train_reviews, test_reviews):
    # para train/test reviews: all reviews to be saved
    train_size = int(len(train_reviews)/2)
    test_size = int(len(test_reviews)/2)
    print("\ninferring embeddings for each document ...")
    train_vectors, train_labels = infer_embedding(model_no, train_reviews, train_size, False)
    test_vectors, test_labels = infer_embedding(model_no, test_reviews, test_size, False)

    print("\nprocessing reviews following their sentiment")
    # for better illustration
    pos_vectors = train_vectors[:train_size] + test_vectors[:test_size]
    neg_vectors = train_vectors[train_size:] + test_vectors[test_size:]

    print("\nsaving embeddings to metadata file ...")
    with smart_open("./results/label.tsv", "w", encoding='utf-8') as f:
        f.write("Index\tLabel\n")
        for i in range(2000):
            label = 0 if i < 1000 else 1
            f.write("%d\t%d\n" % (i, label))
        
    with smart_open("./results/metadata.tsv", "w", encoding='utf-8') as f:
        for i in range(len(pos_vectors)):
            for j in range(len(pos_vectors[i])):
                f.write("%f\t" % pos_vectors[i][j])
            f.write("\n")
        for k in range(len(neg_vectors)):
            for l in range(len(neg_vectors[k])):
                f.write("%f\t" % neg_vectors[k][l])
            f.write("\n")
    print("\nmetadata processing, DONE")
    
