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

Concatenated Doc2Vec models
| index | dm model id | dbow model id | accuracy |
| -- | -- | -- | -- |
| 1 | 15 | 18 | 84.1% |
| 2 | 15 | 20 | 84.4% | 


### note 1
According to Le & Mikolov 2014, PV-DM alone usually works well for most tasks (with state-of-art performance), but its combination with PV-DBOW is usually more consistent across many tasks that they tried and therefore strongly recommended.

In this project, however, contrary to the results of the paper, PV-DBOW alone performs as good as anything else. Concatenating vectors from different models only sometimes offers a tiny predictive improvement - and stays generally close to the best-performing solo model included.

### note 2
re-inferring doc-vectors
Because the bulk-trained vectors has much of their training early, when the model itself was still settling, it is sometimes the case that rather than using the bulk-trained vectors, new vectors re-inferred from the final state of the model serve better as data for downstream tasks.