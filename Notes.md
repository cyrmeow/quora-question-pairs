# Quora Question Pair & Getting Started with Kaggle

## Notebooks

Throw away the built-in python version in macOX, install another python and use `pip install jupyter` to install jupyter to enable myself to look into others' notebooks XD

And that cost me 2 hours :(

## TF-IDF

### Document Frequency

- Rare terms are more informative than frequent terms - high weights
- Frequent terms are less informative - positive weights

### idf weight

$df_t$ is the document frequency of t: the number of documents that contains t, which is an inverse measure of the informativeness of a term $t$.
$idf_t$ is defined by: 
$$idf_t = \log_{10} (N/df_t)$$

### tf-idf weighting

The tf-idf weight of a term is defined by
$$w_{t,d} = (1+\log tf_{t,d}) \times \log(\frac{N}{df_t})$$
The tf-idf weight increases with the number of occurrences within a document, and increases with the rarity of the term in the collection

### Score of a Document for Ranking
With tf-idf weighting, the final score of a document for a query is defined by
$$Score(q,d) = \sum_{t \in q \cap d} tf-idf_{t,d}$$