# sentiment-detection


We investigate the use of document embeddings (```doc2vec```) in the sentiment classification problem, or more specifically, categorising movie reviews. Since there are a number of hyperparameters in ```doc2vec``` models, we evaluate every one of them and then the optimal hyperparameter setting. After tuning hyperparameters of Support Vector Machine (SVM) via a 10-fold cross-validation, we examine the performance of SVM classifier using ```doc2vec``` in comparison with the baseline system, a Naive Bayes (NB) classifier using Bag-Of-Words (BOW) features. We finally analyse the embedding space, which is learnt in ```doc2vec``` models, by visualizing similar subgroups of documents and inferring similar words given the target word. 

We found that Support Vector Machine based classifiers using document embeddings (89.35\%) significantly outperform Naive Bayes classifier using traditional Bag-Of-Words representations (82.25\%).

## Naive Bayes

The Naive Bayes (**NB**) classifier is based on the Bayes rule:

$P(c|d) = \frac{P(c) P(d|c)}{P(d)}$

Since NB assumes that every feature $f_i$ is conditionally independent given class $c$, the term $P(c|d)$ can be expressed as:

$P_{NB}(c|d) := \frac{P(c)\prod_{i=1}^{m} P(f_i|c)^{n_i(d)}}{P(d)}$

This is the foundation of NB classifiers and we apply unigrams and bigrams as the features, both of which are basically the bag-of-$n$-words model with different $n$ values. Pang et al. (2002) discussed the use of bigrams might undermine the conditional independence assumptions, and bigrams caused classification accuracy to decline by 5.8% in their experiement, and by 1.4% in our experiment.

## Support Vector Machines

Support Vector Machines (SVM) are considered as a universal learner to solve various kinds of Machine Learning problems, and with the introduction of Bag-Of-Words and the property of being robust on high feature dimensionality, SVM has been proved effective on text categorization tasks. More information could be referred to this [practical](https://github.com/ZihengZZH/machine_learning_practical/tree/master/prac_svm).

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

Full table of doc2vec models pre-trained in the project:

| index | algorithm | vector size | window size | negative samples | hierarchical softmax | epochs | accuracy |
| -- | -- | -- | -- | -- | -- | -- | -- |
| 1 | DM | 100 | 10 | NA | Y | 10/20 | 78.1% / 79.5% |
| 2 | DM | 100 | 10 | 5 | NA | 10/20 | 81.9% / 82.9% |
| 3 | DM | 100 | 20 | 5 | NA | 10/20 | 82.0% / 82.8% |
| 4 | DM | 150 | 10 | NA | Y | 10/20 | 78.6% / 79.9% |
| 5 | DM | 150 | 10 | 5 | NA | 10/20 | 82.4% / __83.9%__ |
| 6 | DM | 150 | 20 | 5 | NA | 10/20 | 81.8% / 82.7% |
| 7 | DBOW | 100 | NA | NA | Y | 10/20 | 86.7% / 87.9% |
| 8 | DBOW | 100 | NA | 5 | NA | 10/20 | 87.8% / __88.2%__ |
| 9 | DBOW | 150 | NA | NA | Y | 10/20 | 86.3% / 86.3% |
| 10 | DBOW | 150 | NA | 5 | NA | 10/20 | 87.3 % / __88.3%__ |

Negative sampling was introduced as an alternative to the most complex hierarchical softmax step at the output layer in doc2vec model, with the authors (Mikolov et al., 2013) finding that not only it is more efficient, but actually produces better word vectors on average. The better performance of negative sampling could be seen from comparison between (1) and (2), (4) and (5), (7) and (8) in the above table.

Concatenated Doc2Vec models

| index | dm model id | dbow model id | accuracy |
| -- | -- | -- | -- |
| 1 | 15 | 18 | 84.1% |
| 2 | 15 | 20 | 84.4% | 



## Comparison between NB classifier and SVM classifier

| index | classifier | features | accuracy (%) | variance (*e-3) |
| -- | -- | -- | -- | -- | 
| (1) | NB | unigrams | 81.15 | .635 |
| (2) | NB | bigrams | 80.85 | .675 |
| (3) | NB | uni/bigrams | 82.25 | .771 |
| (4) | SVM | unigrams | 83.40 | .544 | 
| (5) | SVM | bigrams | 80.90 | *.509* |
| (6) | SVM | uni/bigrams | *84.10* | .709 | 
| (7) | SVM | DM | 83.25 | .476 | 
| (8) | SVM | DBOW | **89.35** | **.345** | 
| (9) | SVM | DM/DBOW | 84.15 | .394 |

where (1) to (6) are the baseline system in our experiment. We can see that SVM using unigrams and bigrams achieve the best preformance in the baseline, and SVM using document embeddings DBOW achieve the overall best performance.

![](https://github.com/ZihengZZH/sentiment-detection/blob/master/results/barchart.png)

**Monte Carlo Permutation Test**

|   | 1 | 2         | 3         | 4     | 5     | 6     | 7     | 8     | 9     |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 
| 1 | - | 1.0  | 0.7780  | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| 2 | 1.0 | -  | 0.6839  | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| 3 | 0.7703 | 0.6709 | -  |  0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| 4 | 0.0002  | 0.0002  | 0.0002  | - | 0.4757 | 0.6863 | 0.8430 | 0.6810 | 0.9198 | 
| 5 | 0.0002  | 0.0002  | 0.0002  | 0.4831 | - | 0.8412 | 0.6845 | 0.8352 | 0.6247 |
| 6 | 0.0002  | 0.0002  | 0.0002  | 0.7040 | 0.8446 | - | 0.9210 | 1.0 | 0.8486 |
| 7 | 0.0002  | 0.0002  | 0.0002  | 0.8384 | 0.6903 | 0.9162 | - | 0.9284 | 1.0 | 
| 8 | 0.0002  | 0.0002  | 0.0002  | 0.6895 | 0.8454 | 1.0 | 0.9240 | - | 0.8340 |
| 9 | 0.0002  | 0.0002  | 0.0002  | 0.9256 | 0.6449 | 0.8418 | 1.0 | 0.8447 | - |

We can see that SVM classifiers all significantly outperform NB classifiers.

### note 1
According to Le & Mikolov 2014, PV-DM alone usually works well for most tasks (with state-of-art performance), but its combination with PV-DBOW is usually more consistent across many tasks that they tried and therefore strongly recommended.

In this project, however, contrary to the results of the paper, PV-DBOW alone performs as good as anything else. Concatenating vectors from different models only sometimes offers a tiny predictive improvement - and stays generally close to the best-performing solo model included.

### note 2
re-inferring doc-vectors
Because the bulk-trained vectors has much of their training early, when the model itself was still settling, it is sometimes the case that rather than using the bulk-trained vectors, new vectors re-inferred from the final state of the model serve better as data for downstream tasks.