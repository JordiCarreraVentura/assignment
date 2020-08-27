

# Clustering

This section describes the clustering pipeline.
1. Section **Problem statement** contextualizes the problem.
1. Section **Assumptions** lists the main premises of the work presented here, including expected input-output schemas and any relevant methodological simplifications.
1. Section **Formalization** provides the high-level overview of our solution to the problem.
1. Section **Findings** briefly summarizes the results.
1. Section **Discussion** provides interpretation of the main highlights from the findings.
1. Section **Running with custom input** provides the instructions on how to use this code on a new dataset.

##	Problem statement

> Cluster disinformation articles into fine grained clusters where the similar articles are in the same cluster.

As per the specifications, this part of the test can be best frame as a clustering problem.

We will approach it as an unsupervised machine learning problem in which input documents must be grouped into semantically homogeneous classes without any prior notion of what the ground truth classes are, or even if they exist at all.

The specific cluster/topic distribution remains a latent variable. It can be modelled as a linear function of the observable variables. In this case, as the observable variables  we will be using standard bag-of-word (BoW) features for simplicity, although more sophisticated dense context-sensitive neural representations like ELMo or BERT embedded-spaces should be used when possible.


##	Assumptions

1. Our pipeline
   1. expects a `TitleTextDataset` object as the input,
   1. performs clustering on the X axis of the dataset object. The axis is of type `list: str` (see variable `X` in `Clusters.py:241`),
   1. ignores the Y axis with the ground truth labels for an unrelated task,
   1. represents each document as a [TFIDF](https://en.wikipedia.org/wiki/Tf–idf)-weighted BoW with some simple pre-processing.
   1. must generate as output a JSON-format file in the same schema as the provided sample (`clusters.json`), namely, an object of type `list: dict`, where each `dict` instance has the following key-value pairs: `"id": URL`, `"text": str`, `"title": str`, `"lang": str`, `"date": datetime`, `"cluster": int`, `"cluster_name": str`. In our case, the schema is simplified into `"id": None`, `"text": str`, `"title": str`, `"lang": "en"`, `"date": None`, `"cluster": int`, `"cluster_name": str` for convenience, as we do not store the original URLs or dates, and the language has been observed to be English in all cases, which eliminates the need for a specific language detection step.
1. We don't know the optimal number of topics in advance, which implies that
   1. [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is not an optimal solution, since it expects to be given the target number of topics,
   1. and neither is [k-means](https://en.wikipedia.org/wiki/K-means_clustering), given that it expects a value for the parameter k, which we want the system to infer for us.
1. We are using a high-dimensional input representation, which implies that
   1. [DB-SCAN](https://en.wikipedia.org/wiki/DBSCAN) is not an optimal solution because its radius parameter has been observed to underperform in high-dimensional spaces,
   1. and neither is [mean-shift](https://en.wikipedia.org/wiki/Mean_shift), for similar reasons.


## Formalization

### BoW representation and pre-processing

### Clustering

The clustering is performed by function `Clusters.clusterize`, which takes the following parameters as input:

| Name | Type | Description |
| --- | --- | --- | 
| `X` | `list: str` | The list of input documents. | 
| `vec` | `sklearn.feature_extraction.text.TfidfVectorizer` | The vectorizer used to transform the input documents into the BoW representation. The vectorizer will be used again during title generation to retrieve the highest-TFIDF n-grams. | 
| `mat` | `scipy`'s sparse matrix object | The object containing BoW-transformed input matrix. |

It follows these steps:
1. Calculate **element-wise similarities** between all documents.
    1. Given that this involves what would normally be a prohibitively expensive _O(n ^ 2)_-complexity loop, the process is sped up using an inverted index.
       1. The index is initialized with the call to function `make_inverted_index` in `Clusters.py:176`, and then looked up for each target document when calling function `lookup_on_index` in `Clusters.py:183`.
       1. The index is word-based and, for any bag-of-words generated by the vectorizer (`vec` parameter), it returns the top-100 documents (parameter `n` in `Clusters.py:143`) that share non-zero-weighted dimensions with the target document (and are also within a 0.3 cosine distance radius of each other, as per the condition in `Clusters.py:150`).
      1. This allows to calculate the element-wise similarity only between documents that are already known to be related, which substantially improves performance and reduces the original _O(n ^ 2)_ complexity back to a value closer to _O(log n)_.
1. Calculate a **minimum cosine similarity** threshold (variable `relevance_threshold` in `Clusters.py:203`) below which matches will be rejected.
     1. For this calculation, we assume that, in any dataset, most documents can be matched to some other document, but also that there will be at least some amount of accidental matches that are the best out of a pool of bad candidates: the resulting match, despite being the best-scoring with respect to the rest of other candidates, might still be a bad match. We find such worst-performers by collecting the similarities of the 5% worst matches for all documents (line `Clusters.py:159`) and averaging the result, then multiplying it by a user-defined constant (parameter `coef` in `Clusters.py:157`). The resulting value is the threshold for any match to be accepted.
1. **Traverse the space** of element-wise similarities and transform each of them into a cluster if it meets the following conditions:
    1. the list of clusters is empty,
    1. the candidate cluster is not subsumed by any of the previously added clusters (this condition is evaluated in `Clusters.py:227`), where subsumption is defined as 34% of the items of the current candidate cluster being included in some other cluster (as implemented in `Clusters.py:121`).


### Title generation

Finally, **cluster names are generated** when serializing the output (which is done in `Clusters.py:255:260`). To do it,
1. document titles are used (as they are likely to contain a good, short paraphrase of the gist of each topic);
1. up to 50 titles are sampled (fewer for smaller clusters);
1. the titles are tokenized and n-grams with _n: n ∈ {1, 2, 3, 4, 5}_  are extracted (`Clusters.py:85.100)`
    1. that don't contain stopwords in n-gram-initial or n-gram final position and
    1. that don't contain non-alphabetical expressions;
1. n-grams are then scored by the sum of the TFIDF of their consistuent tokens times the order of the n-gram (to prefer longer expressions), <img src="https://render.githubusercontent.com/render/math?math=n\ \sum_{i=1}^{|gram|} TFIDF(gram_i)"> ;
1. the highest scoring n-gram is returned as the title;
1. each cluster's title is cached the first time it is generated and re-used by all elements in that cluster to avoid regenerating every time (which was found to greatly hurt performance due to the relatively intensive matrix traversal operations involved).


## Findings

#### Cluster distribution

| Size bucket | # of clusters |
|---|---|
|51-100|3|
|21-50|5|
|11-20|10|
|6-10|5|
|2-5|117|
|1|1180|

#### 18-top clusters

| Size | Title | Comments |
|---|---|---|
| 73 | "conspiracy theory" | |
| 73 | "bill gates behind the genesis" | |
| 72 | "russian doctor reveals the secret" | |
| 33 | "corona virus from space" | |
| 30 | "page is not available" | Data ingestion error |
| 26 | "transfer sk to chinese wuhan" | |
| 23 | null | Data ingestion error |
| 21 | "amir khan" | |
| 20 | "united states behind the outbreak" | |
| 17 | "struck cruise ship" | False positive due to noise in the HTML text |
| 16 | "american biological attack" | |
| 15 | "work to manufacture a vaccine" | |
| 13 | "washington should reveal its role" | |
| 13 | "finland deaths increase" | |
| 12 | "latest news and breaking stories" | |
| 12 | "corona is not considered deadly" | |
| 11 | "new blockade on iran" | |
| 11 | "dangers of developing biological weapons" | |

#### Remarks
1. Almost 40% of the articles could be assigned to some cluster.
1. The data shows a clearly long-tailed distribution.
1. There are 18 comparatively large clusters (11-100 elements), all of which seem significantly semantically consistent.
1. There is a known issue whereby highly similar matches are not added to the right cluster. Due to time constraints, I have not been able to debug the information leak.


## Discussion
1. **Duplicates.** The data contains a large number of duplicates and near-duplicates. There may be a naturalistic explanation, since often many different media outlets essentially copy the same news item from each other, or publish a template they receive from a news agency with little modification. An alternative, more mistrustful explanation is that these particular news items have been pushed across several platforms by an interested agent in order to spread a particular viewpoint on an issue. Both explanations could be true simultaneously.
1. **HTML noise.** The data contains some instances with significant noise due to boilerplate text (cluster `struck cruise ship`). There seems to be room for improving the functionality for extracting text from HTML. This is likely to greatly improve the performance of the clustering process.
1. **Cluster names.** The title-based generation of cluster names seems to yield fluent, informative descriptors.
1. **Re-frame as exploratory data analysis.** From a methodological point of view, the clustering task should have been performed as an exploratory-data-analysis step before the fake news detection task, since cluster information would have been helpful for
    1. detecting boilerplate text that should have been removed from all documents,
    1. detecting duplicates or near-duplicates that should been removed or under-sampled to decrease data imbalance and bias in the dataset (any models trained on the raw dataset are likely to be overfitting to accidental patterns in the data, which would explain the 95+% F-score measurements obtained in the fake news detection task even with Naive Bayes classifiers).


## Running with custom input
The module does not implement yet a command-line interface to be executed on arbitrary inputs from the terminal, but this can be achieved with a simple modification. Assuming a JSON-like input format of type `list: dict` where each dictionary in the list corresponds to a document and has a key `text` with the document's content as the value, and a key `title` with the document's title as the value, e.g.

```
documents = [
	{
		"text": "document text",
		"title": "the document's title",
		"some_other_key": None
		...
	}
]
```
then simply
1. uncomment lines `Clusters.py:242:245`
1. change the path in line `Clusters.py:244` so that it points to JSON with the new dataset
1. run `python3 Clusters.py` on the command line.
