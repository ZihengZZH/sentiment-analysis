# TO DO 
**sentiment detection project**
---

1. ~~improve the structure of the project~~
2. ~~training and testing -- separated~~
3. implementation of the unsmoothing 
   * when the smoothing parameter is missing, the classifier always misclassifies with the accuracy 1.0 and 0.0
4. check the correctness of the classifier
   * ~~either the full vocabulary is built on training set or entire dataset~~
   * ~~normalize the text (lower case)~~
   * the NB classifier is too good when compared to sklearn built-in Gaussian NB classifier (83.5% > 70%), but similar to sklearn built-in Gaussian NB classifier (83.5% ～ 84.1%)
   * While, after 10 folding averaging, the accuracy is only around 80%
5. ~~significance testing -- again~~
6. POS tag -- w/ or w/o could make a difference in NB (Pang2002)
7. ~~SVMLight as the implementation for the practical~~
   * classification accuracy: 80.5% precision: 85.88% recall: 73.0%
8. word2vec implementation in Tensorflow
   * test on samples
9. ~~pre-train several doc2vec embeddings with different parameters~~
    * ~~training algorithm (dm, dbow)~~
    * ~~the size of the feature vectors (100 dimensions)~~
    * ~~number of iterations / epochs (10 or 20)~~
    * ~~context window (10 / 20)~~
    * ~~hierarchical softmax (faster version) (hs == 1)~~
    * ~~save the model into different directories~~
10. ~~use SVM-light classifier with doc2vec embeddings~~
    * SVM + doc2vec classifier accuracy == 88%
11. investigate the hyper-parameters of SVM-Light
    * ~~what kind of SVM is SVM-Light? which kernel?~~
    * ~~How to tune the parameter? GridSearchCV~~ -- very expensive computation
12. cross validation (train/valid/test)
    * ~~alternative in sklearn applied~~
13. ~~permutation test -- need to understand~~
    * ~~implement permutation test~~
14. ~~train the various doc2vec models using the IMDB movie review database~~
    * paragraph embeddings from original dataset refers to this model
15. ~~concatenate doc2vec to infer document embeddings~~
16. SVM performance error analysis
    * which documents does the system make the most catastrophic errors for (and why?)
    * goal: changes in algorithm or parameters to improve results
    * consider a sizable amount of errors and try to classify them
      * likelihood of fixing them
      * frequency of the error
      * source of the error
17. Deployment test
    * ~~choose some IMDB reviews for movies from 2017 or 2018 you liked or disliked~~
    * the data you will test on is then really new, unseen and real
    * the reason why deployment test:
      * wrong assumptions about data (types of films, language)
      * unrepresentative sampling
      * model over- or underfitting
      * taste and fashion over time
18. Insightful analysis of embedding space
    * find out what the model is really doing
    * start from know similar grouping of reviews, then look at their distance in Embedding space
    * similarity must be defined before you measure angles between embeddings
    * recommend https://projector.tensorflow.org
19. Re-structure the project
    1.  classifiers
        1.  NB
        2.  SVM
        3.  RF
    2.  embeddimgs
        1.  BOW
        2.  WORD2VEC
        3.  DOC2VEC
        4.  LSTM
    3. dataset / data loader
20. LSTM-RNN based models
21. Tencent_AILab_ChineseEmbedding
    1.  embeddings / feeding into classifier
    2.  embeddings / feature eng / feeding into classifier

## NOTES
---
The [__glob__](https://docs.python.org/2/library/glob.html) module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order.

In a Doc2Vec object:
* 'hs' parameter (0/1) indicates the hierarchical softmax. If 1, hierarchical softmax will be used for model training. If set to 0, and 'negative' is non-zero, negative sampling will be used.
* 'negative' parameter (int) indicates the negative sampling. If > 0, negative sampling will be used, the int for negative specifies how many 'noise words' should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
* 'min_count' parameter (int) ingores all words with total frequency lower than this
* 'sample' parameter (float) is the threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5)

In the SVM-Light classifier, the output predictions is a list of float, which ranges from -inf to +inf. Each value indicates the prediction of any given test sample, where float > 0 means positive and float < 0 means negative.
