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
5. significance testing -- again
6. POS tag -- w/ or w/o could make a difference in NB (Pang2002)
7. ~~SVMLight as the implementation for the practical~~
   * classification accuracy: 80.5% precision: 85.88% recall: 73.0%
8. word2vec implementation in Tensorflow
   * test on samples
   *  
9. ~~gensim python doc2vec library~~
    * training algorithm (dm, dbow)
    * the size of the feature vectors (100 dimensions)
    * number of iterations / epochs (10 or 20)
    * context window
    * hierarchical softmax (faster version)
10. use SVM-light classifier with doc2vec embedding

## NOTES
---
The [__glob__](https://docs.python.org/2/library/glob.html) module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order.