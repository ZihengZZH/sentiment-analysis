# sentiment-detection

## Naive Bayes

## SVM

## Bag-Of-Words Model
Early state-of-art document representations were based on the bag-of-words model, which represent input documents a fixed-length vector. Bag-of-words models are surprisingly effective but still lose information about word order. Bag of n-grams models consider word phrases of length n to represent documents as fixed-length vectors to capture local word order but suffer from data sparsity and high dimensionality.

## word2vec
learning neural word embeddings (Mikolov et al., 2013)

word2vec is a more recent model that embeds words in a lower-dimensional vector space using a shallow neural network. The result is a set of word-vectors where vectors close together in vector space have similar meanings based on context, and word-vectors distant to each other have differing meanings. 

Two models in word2vec:
* skip-gram model
* continuouts-bag-of-words model



## doc2vec
embeddings for sequences of words (Le and Mikolov, 2014)

doc2vec acts as if a document has another floating word-like vector, which contributes to all training predictions, and is updated like other word-vectors, but we will call it a doc-vector.

Two models in doc2vec:
* distributed memory (PV-DM)
* distributed bag of words (PV-DBOW)

### note
According to Le & Mikolov 2014, PV-DM alone usually works well for most tasks (with state-of-art performance), but its combination with PV-DBOW is usually more consistent across many tasks that they tried and therefore strongly recommended.

In this project, however, contrary to the results of the paper, PV-DBOW alone performs as good as anything else. Concatenating vectors from different models only sometimes offers a tiny predictive improvement - and stays generally close to the best-performing solo model included.