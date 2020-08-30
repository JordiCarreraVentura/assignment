
#	Misinformation detection

This section describes the clustering pipeline.
1. Section **Problem statement** contextualizes the problem.
1. Section **Assumptions** lists the main premises of the work presented here, including expected input-output schemas and any relevant methodological simplifications.
1. Section **Formalization** provides the high-level overview of our solution to the problem, including subsection **Hypotheses** along with its two headings, **Findings** and **Conclusions**, where we review several experimental set-ups we tested and present their results. The conclusion to this section lies the foundation for our baseline system for the task.
1. Section **Results of the Transformer fine-tuning pipeline** presents the results of our main architecture as introduced in the **Formalization** section.
1. Section **Discussion** goes on to review the results in more depth: it explores the hypothetical causes of the main findings in our experiment, along with their potential implications for production systems, and finally uses them as context to motivate our recommendations for the 6-month plan.
1. Section **Running with custom input** provides the instructions on how to use this code on a new dataset.

## Problem statement

> Your mission is to build a classifier that detects if a given article is real or disinformation and cluster the disinformation articles so the analysts can execute their goals.

Based on the description above, the goal of this task is to implement a classification pipeline that takes documents (strings) as input and returns a binary prediction as output (by convention, we will define _1 = disinformation_ (a **positive** case of fake news), _0 = real_ (a *negative* case of fake news).

For this task, we are provided with a labeled dataset with ground truth training data, which makes it possible to approach it as a **supervised learning** problem.


## Assumptions

1. Same assumptions as for #1 for the **Clusters** task.
1. Following standard practice in document-level prediction tasks, we will assume that word order is orthogonal to the problem across all settings **except** the Transformer-based pipeline, which uses a Transformer architecture as the sentence encoder and, therefore, can leverage word order information implicitly at the sentence level.
    - Consistent with this assumption, we will assume that a bag-of-words (BoW) model will provide a suitable baseline for this task.
    - Similarly, we will assume that misinformation can be modelled as a linear function whereby the frequency of occurrence of a particular subset of terms is directly correlated with the probability that the text is misinformation or, put another way, that there are linguistic expressions typically associated with misinformation, e.g. if a text uses the 3-gram "vaccines cause autism", it is likely to be misinformation.
1. We will ignore several potential non-linear phenomena involved in machine reading comprehension and Natural Language Understanding applications that can be expected to have an impact on this task. More specifically, we will not be handling **negation** or **quotes** explicitly . This is likely an over-simplification, as e.g. a pro-abortion post focused on debunking an anti-abortion piece may quote (or refute/negate) a lot of excerpts from the latter and, therefore, end up more highly correlated (in a superficial sense) with the variables in the misinformation class than the real news class.
1. We will assume the dataset is balanced.
    - An exploratory data analysis of the training and test data revealed that, **while** the main classes are indeed balanced (50% of the documents belong to the misinformation class, 50% to the real news dataset), almost 40% of the documents are inside a larger cluster whose members are near-duplicates of each other.


## Formalization

### Proposed model

As the main pipeline, we will stick to the industry state-of-the-art and use a transfer learning pipeline combining a pre-trained model with a fine-tuning step.

As the **pre-trained model**, we will be using Google's [Universal Sentence Encoder (USE)](https://arxiv.org/pdf/1803.11175.pdf) as a departure from the more widespread standard practice of using a BERT or XLNet variant to provide the pre-training. The main reason for our choice is the fact that it the USE is a cross-lingual embedding space and provides built-in multilingual support. This allows the model to handle several languages simultaneously without additional modifications. Although this is not an explicit requirement of this task, it is usually a commercial desideratum, as it would allow our model to handle English inputs and Spanish or Italian inputs indistinctively, which seems desirable as misinformation is obviously not restricted to a particular language. In particular, the USE is a

1. [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf, attention-based neural networks,
1. [sentence encoder](https://arxiv.org/pdf/1803.11175.pdf),
1. [cross-lingual word embedding space](https://pdfs.semanticscholar.org/c44b/c8995403807414619fc135d9ba87a447bcaf.pdf),
1. [knowledge distilled model](https://arxiv.org/pdf/1910.01108.pdf).

These concepts are briefly summarized below for clarity:

- **Transformer** A neural network that removes the mechanism of recurrence and uses several attention layers instead for the purpose of modelling short- and long-distance relations between the words of a document.
- **Sentence encoder** A type of architecture that generates sentence embeddings from word embeddings by combining a Transformer with some other form of modelling beyond simple average- or maximum-pooling of the word embeddings themselves. In the case of Google's USE, a Deep Averaging Network (DAN) is stacked on top of a Transformer; it takes as input the averaged unigram and bigram embeddings from the Transformer and uses them as input for a single feed-forward layer. The stack is then trained with a multi-task objective.
- **Cross-lingual word embedding space** A semantic vector space model where translations into several languages are all mapped onto similar regions of the shared space, effectively creating an embedding for the same abstract concept across all the languages regardless of its specific translation into each of them. This allows for effective language independence and enables the model to calculate semantic distance measurements between words in either language combination. This, in turn, makes it possible to fine-tune the model on task-specific data for any of the languages and simultaneously learn to solve the same task in any of the other languages, geometrically decreasing the need for training data while at the same time solving all data-sparsity related problems typically associated with low-resource languages.
- **Knowledge Distillation** Distillation consists in training a model with the same **inputs** that were used to train a much larger model, **as well as the outputs** from that larger model. It has been shown that the smaller model can learn to replicate the results of the larger model within 98-99% of its performance while being many times smaller and accordingly more computationally efficient. The intuition is that the complexity of the larger model is necessary to more reliably avoid local optima during training (while only partial information is available) but that, once the model has successfully converged towards a global optimum, the latter can be encoded in a much more compact model.

For the fine-tuning step, we will simply use the encoded document-level embeddings to train a standard supervised classifier.

To generate representations at the document level, the Transformer-encoded sentence embeddings will have to be aggregated (ideally while keeping the same dimensionality), which in our case will be done using one out of three possible types of pooling post-processors (average, median, or maximum pooling) or defaulting to running the encoder on the whole document text as opposed to on a per-sentence basis.


### Baseline

As the baseline, and based on Assumptions.2 and Assumptions.3, we will resort to a traditional BoW model. To inform its definition in a more empirical way, as well as to get a better preliminary understanding of the task, we will test several hypotheses. Any hypotheses that we can validate will be adopted as the configuration for this model. We also expect this hypotheses to provide some insight as to how to best formalize the problem, and hence inform our final proposal of an optimal architecture for the 6-month plan.


### Hypotheses

To create a competitive baseline as well as to find the most suitable encoder for the neural pipeline, we explored several hypotheses.

Below is a table with the overview.

In the next section, **Findings** we detail our findings, and in the section below it, **Conclusions**, we outline the baseline configuration.

| Hypothesis ID | Claim | Code |
|---|---|---|
| 1 | Pre-processing improves performance | Pipeline: `hypotheses.py:52:94`  <br> Parameter grid: `params.py:43:84` |
| 2 | Title information improves performance | Pipeline: `hypotheses.py:99:121`  | 
| 3 | n-gram features improve performance | Pipeline: `hypotheses.py:126:148`  <br> Parameter grid: `params.py:91:123` |
| 4 | Character n-grams improve performance | Pipeline: `hypotheses.py:153:175`  <br> Parameter grid: `params.py:126:164` |
| 5 | There is an optimal hyper-parameter configuration for this task | Pipeline: `hypotheses.py:180:213`  <br> Parameter grid: `params.py:170:239` |
| 6 | N/A | N/A |
| 7 | Encoder choice has an impact on Transformer performance | Pipeline: `hypotheses.py:247:268`  |
| 8 | Stacking a classifier on top of a feature selection step improves performance | Pipeline: `hypotheses.py:273:300`  <br> Parameter grid: `params.py:242:286` |
| 9 |  The size of the fine-tuning dataset directly correlates with model performance | Hyper-parameter `k` in `TransformerPipeline.py:103:148` |

#### Findings

1. **Hypothesis #1**
   1. was refuted generally speaking: on the held-out dataset, the model using input pre-processing had virtually the same performance as the baseline model across all settings;
   1. more interestingly, it was not just refuted but actually contradicted during the cross-validation training stage, as shown in TABLE_FEXT. More specifically,
       1. removing entities caused a drop in performance across all settings (expected, as some entities are likely to be informative, yet still sub-optimal, since we would like the model to learn general expressions as predictors in order to avoid overfitting to particular news stories);
       1. removing stopwords caused the second largest drop in performance (also a troubling sign, as it suggests the model might be assigning weights to class-independent features and, again, overfitting to potential bias towards statistically anomalous distributions of function words, e.g. in a story about somebody spying _for_ somebody else, the preposition _for_ may actually occur more often than chance and that could be causing the model to give it a weight. We observed this phenomenon with high-frequency noise in one of the clusters, involving page ads and spam stories about e.g. Turbo tax despite the fact that the story was about a cruiser's situation during the beginning of the Covid-19 pandemic);
       1. removing non-linguistic expressions had a mixed effect (showing that numerals were probably not being used as features by the model anyway).
   1. Points 1.2 and 1.3 are consistent with overfitting: both the model with and the model without pre-processing had the same performance on the held-out test data, suggesting that any additional features captured by the former did not generalize (and had actually no impact) when predicting on unseen data, suggesting that none of the pre-processing heuristics removes important information. It is still interesting that the cross-validation did not reflect this, pointing at the limitations of an intrinsic evaluation setting.
1. **Hypothesis #2**
   1. was partially confirmed: there was a slight 1% macro-F1 improvement for Logistic Regression and Naive Bayes classifiers, and no effect for SVMs,
   1. unsurprisingly, title-only settings were the worst-performing
   1. yet, surprisingly, they still reached 88% macro-averaged F1. Although document features still amount to a 10% improvement and should therefore be used in all cases, it is puzzling that the titles alone contain enough information for the model to approach 90% F1. This implies that titles are exceptionally good summaries of the documents, and that whatever information can be captured in a 10-12-words' worth of language data, is largely the same information leveraged by the model to make its decisions. Since titles usually do not offer a lot of nuance or commentary, but provide instead of succint summary of the main idea of a document, this result suggests that the model is basing its predictions on a small set of expressions and entities rather than a broad understanding of the vocabulary of a category.
1. **Hypothesis #3** was partially confirmed for discriminative models (table TABLE_GRAMS): it had a minor impact (+1% F1) on logistic regression and SVMs using n-grams with _n = 3_, but it actually hurt Naive Bayes (almost 6% drop in the same setting).
1. **Hypothesis #4** was refuted in that character n-grams did not improve performance, but they also did not hurt it, which means that it may be viable to use them as the input representation if we wanted to decrease the model size (the dimensionality of character n-gram models is substantially smaller than in word n-grams) or if we wanted to increase the model's generalization capabilities via subword processing. Based on our experiments, there is currently no such need, which is we defaulted to word n-grams for the higher training speeds.
1. **Hypothesis #5** was largely refuted for the parameter settings we considered, but the search was not exhaustive due to time limitations so we will not discuss these results in more depth.
1. **Hypothesis #6** had to be dropped due to time constraints.
1. **Hypothesis #7** was confirmed: average- and median-pooling yielded better results on average, suggesting that they produce richer document representations.
1. **Hypothesis #8** was refuted: although there was a 1.5%-2% improvement in F1 for any of the stacks combining SVMs and logistic regression (with either as the feature selector or as the classifier), it seems possible to trace back the improvement to the use of n-gram features.
1. **Hypothesis #9** was confirmed: the size of the fine-tuning data was found to be a strong predictor of model performance. Refer to Results.2 for more a longer discussion.

##### TABLE_FEXT

**SVM**

|param_fext__remove_entities|param_fext__remove_nonalpha|param_fext__remove_stopwords|mean_test_score|rank_test_score|
|---|---|---|---|---|
|False|False|False|0.9738|1|
|False|True|False|0.9719|2|
|False|True|True|0.9712|3|
|False|False|True|0.9712|3|
|True|False|False|0.9694|5|
|True|False|True|0.9675|6|
|True|True|False|0.9669|7|
|True|True|True|0.965|8|

**Naive Bayes**

|param_fext__remove_entities|param_fext__remove_nonalpha|param_fext__remove_stopwords|mean_test_score|rank_test_score|
|---|---|---|---|---|
|False|False|False|0.9625|1|
|False|True|False|0.9613|2|
|False|True|True|0.96|3|
|False|False|True|0.9594|4|
|True|False|False|0.9531|5|
|True|False|True|0.9525|6|
|True|True|True|0.9519|7|
|True|True|False|0.9519|7|

##### TABLE_GRAMS

**Naive Bayes**

|param_vec__ngram_range|mean_test_score|rank_test_score|
|---|---|---|
|(1, 1)|0.9625|1|
|(1, 1)|0.9625|1|
|(1, 1)|0.9625|1|
|(1, 2)|0.9581|4|
|(1, 2)|0.9563|5|
|(1, 2)|0.9556|6|
|(1, 1)|0.955|7|
|(1, 1)|0.955|7|
|(1, 1)|0.955|7|
|(1, 3)|0.9419|10|
|(1, 3)|0.9406|11|
|(1, 2)|0.9356|12|
|(1, 3)|0.9337|13|
|(1, 2)|0.9325|14|
|(1, 2)|0.9281|15|
|(1, 3)|0.9131|16|
|(1, 3)|0.9094|17|
|(1, 3)|0.9031|18|


**SVM**

|param_vec__ngram_range|mean_test_score|rank_test_score|
|---|---|---|
|(1, 3)|0.9819|1|
|(1, 2)|0.9781|2|
|(1, 3)|0.9775|3|
|(1, 3)|0.9775|3|
|(1, 2)|0.9769|5|
|(1, 2)|0.9769|5|
|(1, 1)|0.9744|7|
|(1, 1)|0.9744|7|
|(1, 1)|0.9744|7|


**Logistic regression**

|param_vec__ngram_range|mean_test_score|rank_test_score|
|---|---|---|
|(1, 3)|0.9637|1|
|(1, 2)|0.9637|1|
|(1, 3)|0.9637|1|
|(1, 3)|0.9631|4|
|(1, 2)|0.9625|5|
|(1, 2)|0.9625|5|
|(1, 1)|0.9587|7|
|(1, 1)|0.9587|7|
|(1, 1)|0.9587|7|

#### Conclusions

As the conclusion for this section, a strong baseline for this task seems to be
1. a margin-based linear classifier like an SVM,
1. using n-gram word features for $n: 1 ≤ n ≤ 3$,
    1. optionally using character n-grams for increased recall,
1. with input pre-processing to remove potential confounders
    1. stopwords
    1. specific entities
1. taking both documents and their titles as input.

In our case, this model is `models/ngrams.TitleTextDataset.svm_ngrams.p`. See section **Running with custom input** below for details on how to use it.



## Results of the Transformer fine-tuning pipeline

Based on the **top-5 entries** in Table SENTENCE_ENCODERS below, we can draw the following conclusions:
1. All five entries are within 2% F1 (93% for the best performing setting, 91% for the 5th), which means their performance is roughly equivalent.
2. The top predictor of high performance is hyper-parameter `k`, which corresponds to the size of the fine-tuning dataset fed to the neural sentence encoder.
    - This result is hardly surprising, as the availability of fine-tuning data is expected to improve the performance of a model in a transfer learning setting, but it is interesting to notice the difference between _k = 150_ and _k = 400_, with the latter significantly improving over the former. Although 400 instances cannot be seen as a large amount of data objectively speaking, transfer learning models have been proven to reach state-of-the-art performance on domain-specific tasks like sentiment analysis or intent detection with as few as dozens of examples. By the same logic, we would have expected our model to plateau earlier (around 150 instances) rather than to keep improving (with 400 instances). This suggests the objective may be particularly hard for the model to converge to. An obvious next step would be to replicate this methodology with _k = 2000_ (the whole task-specific training set).
3. The 2nd top predictor is the classifier, with SVMs taking 3/5 of the spots, and logistic regression classifiers the other 2.
   - The comparatively lower performance of Random Forests is intriguing, given that they can model significantly more complex functions. In addition, Random Forests are more robust than Decision Trees, which also mitigates the risk of overfitting. If we accept these premises, there is a possibility that Random Forests may actually be showing the actual performance we can expect from our model in this task, and would imply that the rest of the models are overfitting. We will go back to this point in the **Discussion** section.
3. The document-level sentence-pooling mechanism seems to have an impact, albeit smaller, with `SentenceLevelAveragePooler` and `SentenceLevelMedianPooler` both tied for the top spot and twice as likely to appear among the top-5 configurations as the maximum pooler.
   - It is possible that maximum pooling over all the sentences of a whole document gave rise to "wobbly" encodings whereby highly-weighted outliers ended up populating most dimensions of the output vector, resulting in an encoding that is not representative of the document's distributional center.
1. Top macro-averaged F1 for the neural baseline on the held-out test dataset was 91.65%, which is significantly lower than any of the baselines, found to be quite competitive despite their simplicity (e.g. 96% macro-averaged F1 for the top-5 stack models and the top-3 baseline configurations, 97.6% for n-gram BoWs, 99% F1 for the best stack model).
   - It is worth noting, though, that baseline performance seems suspiciously high and suggests potential overfitting. The neural pipeline's results are more realistic, and still high given the difficulty of the task. We will address this issue in greater depth in the **Discussion** section.


#### SENTENCE_ENCODERS

|k|classifier|mean_test_score|param_encoder|
|---|---|---|---|
|400|LinearSVC|0.93|SentenceLevelAveragePooler|
|400|LinearSVC|0.92|SentenceLevelMedianPooler|
|400|LogisticRegression|0.92|SentenceLevelAveragePooler|
|400|LinearSVC|0.91|SentenceLevelMaximumPooler|
|400|LogisticRegression|0.91|SentenceLevelMedianPooler|
|400|LinearSVC|0.9|StringConcatenationEncoder|
|50|LinearSVC|0.88|SentenceLevelAveragePooler|
|50|LinearSVC|0.88|SentenceLevelMedianPooler|
|400|LogisticRegression|0.88|StringConcatenationEncoder|
|15|RandomForestClassifier|0.87|SentenceLevelMaximumPooler|
|150|LinearSVC|0.87|SentenceLevelMedianPooler|
|150|LinearSVC|0.86|StringConcatenationEncoder|
|400|LogisticRegression|0.85|SentenceLevelMaximumPooler|
|400|RandomForestClassifier|0.85|SentenceLevelAveragePooler|
|150|LinearSVC|0.85|SentenceLevelAveragePooler|
|150|LogisticRegression|0.83|SentenceLevelAveragePooler|
|150|LogisticRegression|0.83|StringConcatenationEncoder|
|150|LogisticRegression|0.83|SentenceLevelMedianPooler|
|150|LinearSVC|0.83|SentenceLevelMaximumPooler|
|400|RandomForestClassifier|0.82|SentenceLevelMedianPooler|
|50|LinearSVC|0.82|SentenceLevelMaximumPooler|
|400|RandomForestClassifier|0.8|SentenceLevelMaximumPooler|
|15|LogisticRegression|0.8|SentenceLevelMaximumPooler|
|15|LinearSVC|0.8|SentenceLevelAveragePooler|
|15|LinearSVC|0.8|SentenceLevelMaximumPooler|
|15|LinearSVC|0.8|SentenceLevelMedianPooler|
|15|RandomForestClassifier|0.8|SentenceLevelAveragePooler|
|50|LinearSVC|0.78|StringConcatenationEncoder|
|50|RandomForestClassifier|0.78|SentenceLevelAveragePooler|
|150|LogisticRegression|0.78|SentenceLevelMaximumPooler|
|150|RandomForestClassifier|0.77|StringConcatenationEncoder|
|400|RandomForestClassifier|0.77|StringConcatenationEncoder|
|50|LogisticRegression|0.76|SentenceLevelMaximumPooler|
|150|RandomForestClassifier|0.76|SentenceLevelAveragePooler|
|50|RandomForestClassifier|0.74|SentenceLevelMaximumPooler|
|50|RandomForestClassifier|0.74|StringConcatenationEncoder|
|15|LinearSVC|0.73|StringConcatenationEncoder|
|15|RandomForestClassifier|0.73|SentenceLevelMedianPooler|
|150|RandomForestClassifier|0.7|SentenceLevelMaximumPooler|
|50|LogisticRegression|0.68|StringConcatenationEncoder|
|50|RandomForestClassifier|0.68|SentenceLevelMedianPooler|
|150|RandomForestClassifier|0.68|SentenceLevelMedianPooler|
|50|LogisticRegression|0.64|SentenceLevelMedianPooler|
|50|LogisticRegression|0.62|SentenceLevelAveragePooler|
|15|LogisticRegression|0.6|SentenceLevelAveragePooler|
|15|LogisticRegression|0.6|SentenceLevelMedianPooler|
|15|LogisticRegression|0.6|StringConcatenationEncoder|
|15|RandomForestClassifier|0.53|StringConcatenationEncoder|



## Discussion
1. **Data imbalance**. As pointed out in Assumptions.4, the balanced-data assumption does not fully hold for the training data, where numerous large clusters could be found.
    1. These clusters could be causing sampling bias in that, if many of their members accidentally share an expression, that expression might end up as a high-frequency predictor of whichever class the cluster happens to be associated with, turning into potentially powerful counfounders.
    1. There is also the risk that, for a really large clusters, their members are present on both the training and test set, resulting in misleadingly high performance in certain settings (very likely in cross-validation pipelines, and highly likely in online prediction settings if models are retrained too often: earlier members of a cluster may pollute the training data for an upcoming iteration of the model and result in a spurious performance boost when predicting on other members of the cluster arriving later).
    1. To address both issues, the ideal pipeline should undersample large clusters down to a representative size, and preferably to single documents. Although an issue (e.g. anti-vaccines, abortion, etc.) should actually be modelled at a topic level on the basis of a large collection of documents, no single specific story/narrative inside that overarching topic should amount to a cluster so large as to determine the classification of the whole topic. Based on some of the symptoms discussed above (Findings.Hypothesis.1.2.1, Findings.Hypothesis.1.2.2, Findings.Hypothesis.1.3, Findings.Hypothesis.2.3, Results.5), we believe that this may actually be happening when considering the current training dataset only.
   1. Otherwise, there is the risk that the model is simply memorizing accidental linguistic patterns in the clusters and exploiting them during prediction, e.g. there might be a large cluster corresponding to a popular news story on vaccines that has garnered a lot of interest and can be uniquely identified by a highly-specific pattern such as a the name of a victim or the place where the events happened, yet both such features are irrelevant for modelling the concept of misinformation. The resulting model will hardly generalize to new data, although it might be a good detector of that particular topic/story.
1. **Competitive baseline**. All BoW baselines were found to be strong on the available data, with performances in the high nineties (measured in macro-averaged F1) compared to a low-nineties performance for the neural pipeline.
    1. However, as discussed in the preceding point, rather than seeing such high performance as merit of the BoW baselines, we take it as sign of potential overfitting: the model may be learning to recognize stories (narratives) that are known to be misinformation, but failing to capture the abstract notion of misinformation.
    1. The former behavior might actually be preferable if we could assume the availability of a large, up-to-date database of story-level training data as opposed to topic-level training data, since that would transfer the burden of the task to the knowledge base. In this scenario, the detection task could be simplified into a problem of determining if an input document matches any of the topics in the database (standard document classification). However, this also requires pro-active maintenance of the knowledge base, which is often expensive.
    1. The alternative would be to ensure that the model is capable of generalizing beyond the story-level and that it models the problem at the topic level. Take for instance the case of meddling into foreign elections: a story-level model could learn to classify new instances of this narrative by the names of specific people involved, rather than mainly on the basis of topical expressions like _government interference_, _voting machines_ and _propaganda_. The first model would show a high performance on particular news items, but would fail to generalize to stories, even when discussing the same topic (election interference). To ensure generalizability, the training data should not focus on any particular story; instead, it should include a representative sample of stories, so that the model can generalize to any commonalities they share.
   1. Note that the score of the Transformer-based pipeline is substantially lower (-6% with respect to the BoW models) yet still quite high. Our conjecture is that, thanks to the pre-training, the model is more robust against orthogonal features (story-level ones, which can be expected to be so rare over the data sizes used to pre-train industry-scale neural language models that they will be uncorrelated from topic-level variables).
1. **Title information**. As already pointed out in Findings.2, document-title information (without the actual document text) was enough to achieve 88% F1, proving that most of the variance of this dataset can be captured by the relatively small set of features occurring in the titles. The fact that this is possible without incurring in underfitting constitutes a strong indication that the problem solved by our model is simple, which contradicts the plausible prior expectation that the problem of detecting misinformation should be hard. Therefore, the problem that we think the system is solving may not be the problem it is actually solving: while we are interested in misinformation prediction, the model actually seems to perform misinformation detection, that is, merely recognize new cases of known instances of misinformation, rather than recognizing new instances of misinformation.
1. **SVM**. Support vector machines ended up as the top performing model across virtually all settings, whether for feature selection, standard supervised classification, or fine-tuning.
1. **Fine-tuning size** was found to be the main predictor of high performance (Findings.8) and the fact that hundreds of examples were needed, as opposed to dozens, suggests that this task is significantly more difficult than other conventional applications like intent detection.
1. In terms of **pooling mechanisms**, maximum pooling was found to lead to unstable representations, with average and median pooling performing slightly better. The question of how to best represent the content of a document remains elusive, though, with no clear neural alternative to BoW-representations for document-classification, despite recent efforts involving Hierarchical Attention Networks and document-level CNNs, or the simpler `doc2vec` representation (which, however, has been proven to perform worse than BoW-based classifiers on standard benchmarks).
1. **Negation handling**. As stated in Assumptions.3, in this experiment we have ignored linguistic phenomena that introduce non-linearities, such as **negation** or **quotes**. Ideally, these should be explicitly addressed by the model, given that they are relatively frequent (the former is very frequent in general, whereas the latter can be highly frequent on a domain basis, e.g. news articles). Crucially, a model trained on examples containing the 5-gram _vaccines cause autism_ should be robust to the occurrence of an instance like _no vaccines cause autism_.
1. **Quote handling**. Similar to the preceding point, a document should be classified on the basis of the claims in that document, rather than on the basis of reported claims that may be have been added to it for context or illustration. For instance, a piece criticizing a conspiracy theory is likely to paraphrase some of the beliefs associated with it before proceeding to refute it, yet those mentions should not be mistaken for actual mentions of the conspiracy theory. The text is about the conspiracy, but it does not advocate for or sponsor it in any way. This could turn out to be a major confounder factor if we expect the model to pinpoint actual pieces of misinformation as opposed to texts reporting on misinformation, or that talk about topics actively targeted by misinformation efforts.
    1. Proper handling of these cases would probably require argument detection technology, or at the very least a stance detection model that can recognize the polarity of a document towards a target subject.



## Running the code with custom input
The script `cli.py` in the repository can be used to get predictions on arbitrary input with any of the serialized models. Contrary to what the name suggests, the script is not really a command-line interface (CLI). However, it can be used effectively with minor changes. Assuming a JSON-like input format of type list: dict where each dictionary in the list corresponds to a document and has a key text with the document's content as the value, and a key title with the document's title as the value, e.g.

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

then the model can be used following the sequence of steps below:
1. download the serialized models
    1. [top-performing BoW model (SVM with n-gram features)](https://www.dropbox.com/s/p9ijlvq7jkuomp6/ngrams.TitleTextDataset.svm_ngrams.p?dl=0)
    1. [top-performing neural document-encoder pipeline](https://www.dropbox.com/s/lm7pepe8yxne2s5/sentence_transformer__sentence_encoders.p?dl=0)
1. comment or uncomment lines `cli.py:28:29` so that the `model_path` variable points to the intended model,
1. run the script with `$ python3 cli.py`






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
1. n-grams are then scored by the sum of the TFIDF of their consistuent tokens times the order of the n-gram (to prefer longer expressions). More formally, an n-gram's score is defined as <img src="https://render.githubusercontent.com/render/math?math=score(gram) = n\ \sum_{i=1}^{|gram|} TFIDF(gram_i)"> ;
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
