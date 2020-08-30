
# 6-month plan

> The analysts know that this will not be the last attack. Thus you are tasked with proposing a 6-month project plan to them on how to detect disinformation campaigns at scale.... Make sure to describe:
> - General steps taken to build the state-of-the-art detection solution
> - Its overall architecture (we also recommend an architecture sketch/drawing)
> - Possible detection models and networks used

For these three desiderata, refer to section **Full system specifications** below and table **Architecture** more specifically.

> - Approach taken with data i.e. what data is needed and from where, how it should be cleaned, annotated, etc.

Covered in sections **Full system specifications** and **Model explainability**.

> - How the solution should be evaluated, made explainable, shipped to production and scaled

Evaluation is covered in table **Architecture**.
Explainability is discussed in depth in section **Model explainability**.
Shipping and scaling is addressed in section **Deployment and scalability**.

> - How will the solution constantly improve
Briefly addressed in **Model explainability.4.3**.
For all other purposes (steps 8-9 in **Architecture** most crucially, but also steps 3-4 to a lesser extent), we just assume an ongoing data annotation effort whereby a dedicated team of experts and linguists work together to keep the datasets update and regularly contribute documents representative of controversial issues and misinformation campaigns in current affairs.


## Full system specifications

Given the following definitions

$d = \text{dimensionality of the output vectors returned by the sentence encoder}$
$D = \text{|input documents|}$
$M = |D|$
$S = max_{i}^{M} |D_{i}|, or the length of the longest document-level sequence supported by the model, that is, the longest document (measured in sentences) in dataset D.$
$W = max_{i}^{M} max_{j}^{|D_i|} |D_{i_j}|, or the length of the longest sentence-level sequence supported by the model, that is, as the longest sentence (measured in tokens) in any of the documents in dataset D.$
$V = \bigcup_{i}^{|D|} \{w: w\text{ is a spaCy token}\wedge  w\in D_i\}$
$C =\text{an array of cluster labels }c: c \in C \wedge |C| = |D|\ \wedge |\{C\}| < |\{D\}|$
$L = \{\text{misinformation issues/topics targeted by the detection pipeline}\}$
$\lambda = |L|$

we would have the following architecture:

**Architecture**

| Step # | Step name | Input | Output | Processor | Metrics |
| --- | --- | --- | --- | --- | --- |
| 1 | Ingestion | `list: str` | `list: spacy.tokens.Doc` | A `spacy` pipeline. | Number of exceptions upon initializing `Doc` objects.<br>Daily volume.<br>Memory usage.<br>Average ingestion speed. |
| 2 | Embedding look-up | `list: spacy.tokens.Doc` | `list: spacy.tokens.Doc` (with `Token` objects mapped to their word embeddings) | A `spacy` pipeline or an extension thereof. | Memory usage. <br>Tokens per second. |
| 3 | Segmentation | `list: spacy.tokens.Doc` | `list: spacy.tokens.Doc` (with `Token` objects annotated with their sentence tags in BIO format) | A custom sequence tagging algorithm, e.g. HMM, CRF, LSTM-CRF or GRU. | F1 at the token level. <br> (Omitting general performance metrics from the previous steps, but they would also apply here.) |
| 4 | Modality detection | `list: spacy.tokens.Doc` | `list: spacy.tokens.Doc` (with `Token` objects annotated with their modality tags in BIO format; several layers may apply: negation, quote, reported speech, hypothetical, and so on) | Same as for segmentation. | Same as above. |
| 5 | Sentence encoding | `list: spacy.tokens.Doc -> M x S x W x d matrix` | `M x S x d matrix` | For the reasons exposed in **Formalization.Proposed model**, we recommend Google's USE or a similar architecture for this component. Alternatively, Google's own BERT or Reformer (a highly optimized Transformer) would constitute valid alternatives. More generally, any other model using knowledge distillation should be preferred (DistilBERT, RoBERTa), given the noticeable increase of performance from this type of models. | Cross-entropy loss of the sentence embeddings with respect to a held-out dataset of domain-specific manually annotated ground truth data. |
| 6 | Document encoding | `M x S x d matrix` | `M x d matrix` | A matrix pooling kernel, preferably average or median pooling, although the concatenation of maximum and averaging pooling should be explored. Alternatively, HANs or CNNs could also be used. Another option would be to stack a document-level LSTM taking sentence embeddings as input on top of the sentence-level Transformer that takes word embeddings as input. Stacked Transformers should probably be ruled out due to both performance reasons and overall model complexity. | Cross-entropy loss of the document embeddings with respect to a held-out dataset of domain-specific manually annotated ground truth data. |
| 7 | Cluster detection | `M x d matrix ∨ M x V TFIDF-BoW matrix` | `((M - N) x d, C) ∧ (N x d, C) mappings ∨ (M - N) x V, C  ∧ N x V, C TFIDF-BoW matrices` | In an online setting, this component would manifest both as<br>1. as a filter based on a classifier that has been trained to recognize previously encountered stories (such that new instances of the same story can be acknowledged as such and handled accordingly: skipped, contextualized, sped through, etc.);<br>1. a clustering algorithm that, given a set of documents that could not be matched to any previously observed cluster, groups them by similarity and generates the array C of cluster tags mapped to axis M of the input matrix. | Cluster size (average, median, minimum, maximum, variance).<br>Cluster homogeneity.<br>Element-wise cosine similarity (average, median, minimum, maximum, variance.<br>(Given that this would normally be an unsupervised machine learning step, it does not seem possible to apply standard Precision, Recall, F1 and Accuracy metrics. An human-in-the-loop Quality Assurance step could be added to that effect but it is beyond the scope of this proposal. |
| 8 | Issue detection | `(N x d, C) matrix` | `(N x (d + 𝛌), C) matrix` | A linear SVM classifier fine-tuned on domain-specific data. Axis C in the input matrix would be informative, as any documents already determined by the clustering algorithm to belong to the same cluster could be grouped together and used as enriched input for the classification, by having any features shared across the whole cluster compound accordingly in the representation received by the classifier. The columns corresponding to term 𝛌 in the output schema will store the multi-label probabilities that a given document talks about each issue being tracked by our system. Note that, in an ideal scenario, we may want to compute this matrix at the sentence level, given that a document is a complex entity containing multiple sentences, specific subsets of which may address different issues even within the same document. For simplicity, at this point we will assume otherwise and will adopt a one-issue-per-document reduction. | Precision, Recall, F1, and Accuracy with respect to a held-out dataset of domain-specific manually annotated ground truth data. |
| 9 | Stance detection | `(N x (d + 𝛌), C) matrix ∧ (list: list: spacy.tokens.Token ∨ N x S x d matrix) ` | `(N x (d + 𝛌), C, P) matrix` | A classifier that takes as input the output of steps 3, 4, 5 and 8, and combines them into stance predictions, e.g. a sentence whose embedding matches the issue detected in step 8 may still be ignored if classified as a hypothetical or as reported speech. Otherwise, sentences semantically aligned with the issue will be taken as positives unless polarity-reversing mechanisms have been detected in them. Thus, if the 3-gram "vaccines cause autism" was found to be significantly more correlated with the misinformation category than the actual news category in the training data, the system will assign it to that category unless e.g. it is preceded by "no" or the relevant tokens have previously been otherwise annotated as polarity-reversed. | Precision, Recall, F1, and Accuracy with respect to a held-out dataset of domain-specific manually annotated ground truth data. |


## Deployment and scalability
Each step/component can be deployed as a separate micro-service using Docker containers.

Containerization essentially guarantees full scalability via horizontal scaling: new instances of any service can be spun automatically on demand with minor supervision.

Given the containerization desideratum, we will also expect each container to be stateless. Any relevant outputs should be stored and centralized in a shared database, ideally a NoSQL one for performance and given that the system will be processing large amounts of raw text and unstructured data.

Containers that depend on trained models should download those models from a central repository upon initialization. Any security-compliant online storage system will be suitable for storing those files.


## Model explainability
1. Regarding the model explainability requirement, some prior considerations apply.
   1. Although traditional architectures involving linear classifiers on top of BoW-based feature extractors are generally viewed as fully- or close-to-fully-transparent white boxes, that can be inspected and debugged almost directly, they cannot readily take advantage of transfer learning, which limits their generalization capabilities and, most importantly, confines them to _ad hoc_ learning on task-specific samples.
      1. High-quality, domain-specific data is not trivial to collect, so this is a substantial limiting factor for these approaches, which must learn from scratch both a model of general language and another model for the specific classification problem. This can imply underfitting for the former and, more often, overfitting for the latter.
      1. The implication for explainability is that, while the system decisions' can be traced back to model weights, and weights can in turn be traced back to specific expressions that the analyst can then interpret and validate semantically, those expressions might actually be orthogonal to the  semantics of the target classes being modelled (e.g. the distinction between story-level features and issue-level features following the terminology we introduced in Discussion.2.3). So, while in this kind of model features can be easily interpreted, they may actually not be interpretable themselves (for semantic rather than technical reasons).
   1. Conversely, state-of-the-art transfer learning architectures can be expected to be more robust to accidental patterns in the data: assume a document contains the n-grams "Roger Stone", "election interference", "Russian" and "hacking". If the training data was small and/or homogeneous enough that no other stories about election interference were included, then from a purely quantitative standpoint our data would entail that e.g. there is no instance of election interference that does not involve Roger Stone (which the analyst, however, would be able to identify as a false generalization). In a transfer learning paradigm, however, the pre-training's priors can be expected to assign higher probabilities to "election interference" and "hacking" in the context of each other, whereas the other terms would effectively be recognized as orthogonal at this stage.
    1. The implication for interpretability is that the encoding obtained as result, while not as immediately accessible for inspection as one directly mapped to a sparse semantic vector space, is however more likely to capture linguistic features that the analyst will find relevant and informative for the task at hand. That is, the neural model's dimensions are probably more interpretable from a semantic point of view, but they remain latent and buried in the parameter space.
2. Given these considerations, the crucial issues for interpretability can be approached from either direction and come down to
   1. how to access neural-quality representations in an explicit way or
   2. how to raise the quality of BoW-based representations up to neural-network levels.
3. With regard to 2.2, the quality of BoW representations could be aligned with that of neural representations by
    3.1. incorporating the notion of pre-training in the BoW vectorization step via model priors,
    3.2. ensuring that the training data for the BoW model is well-balanced both at the target class level (issue-level) and the cluster level (story-level), to control for common sources of lack of representativity such as sampling bias, selection bias, and popularity bias.
    (Note that all these points also apply to the neural approach.)
4. With regard to  2.1,
   1. there are several libraries available for visualizing the activation patterns of neural networks during prediction, which would provide some level of insight into as to which neurons are causing specific system decisions (and those neurons could then be tweaked as needed) but would not still provide full direct traceability down to the level of particular linguistic expressions. The analysts would be afforded some degree of control on the system decisions, but it would be largely reactive and would still rely on strong assumptions;
   2. subject to performance considerations, ablation analysis could be used to reverse-engineer model parameters via ablation of specific input expressions. By measuring the difference between the original and ablated representations, it would be possible to estimate an expression's specific contribution to the model. This process would be computationally expensive so it would only be possible on a subsample of the inputs and, preferably, for a small target vocabulary. Over time, however, it may add up to a significant amount of explicit knowledge about the model's internal states;
   3. system decisions could also be contextualized via exemplification, by retrieving k-nearest neighbors from the semantic vector space model: once a model has been fine-tuned, it is possible to determine which dimensions went on to become the top predictors. An efficient [k-nearest-neighbor search algorithm](https://github.com/spotify/annoy) could then be used to retrieve pre-training embeddings that share high values for the same dimensions. The resulting set of neighbors should provide a relatively clear illustration of what was the semantic representation determining the system's decisions;
    4. finally, [ExBERT](https://arxiv.org/pdf/2005.01932.pdf) can be used for top-down feature engineering on neural representations, giving analysts a high degree of control over the system's behavior while retaining the core advantages of transfer learning models.
       1. ExBERT fine-tunes BERT on a logical entailment task (the MultiNLI dataset), then uses the resulting inference model to create a separate encoding for each rule specified by the analysts and, finally, concatenates the encodings for all rules as the input representation for a classifier. In other words, ExBERT first fine-tunes BERT for entailment detection, then uses the resulting model as a few-shot learner over manually defined rules stated as additional natural language input (rather than domain-specific data) and, finally, instead of replacing the original weights with the fine-tuned ones, the latter are concatenated as additional dimensions, so that each rule's contribution to the system decision is fully traceable.
       2. By doing so, BERT's implicit knowledge is being made explicit along dimensions pre-defined by the analyst as relevant, which allows for explicit access to implicit knowledge, and greatly improves the interpretability of the model. In essence, it performs an ablative analysis like 4.2 but at the parameter level, rather than the input level. As an example,
         - given a text _t_ expressing an anti-vaccination stance, 
         - and given the following hypothetical set of rules R:
            1. _autism has a genetic cause_
            2. _autism has environmental triggers_
            3. _most people are vaccinated_
            4. _few people are autistic_
            5. _vaccines are tested_
            6. _vaccines are regulated_
         - ExBERT separately encodes each pair _<t, r>_ for r: ∈ R and concatenates all the output vectors;
         - such that
            1. any simultaneous activations for the set of columns corresponding to the 4th and 5th rules would likely correspond to a negative document (not misinformation) that adheres to the general public opinion on the issue and resorts to well-established lines of argumentation to that effect;
            2. any documents that do not activate any of these sets of dimensions would be either a true positive (a document about the controversy that does not adhere to any of the standard arguments) or a false positive presenting an original, previously-unobserved point of view (which should probably be added as a further rule to the ExBERT model).
       3. This workflow would allow for a reasonable compromise between model extensibility (adding rules is easier than adding collections of documents, as per Discussion.5), generalization (adding embedding-based features is better than adding token-level features), and timeliness (the model would be able to recognize cases of relevant documents not fully accounted for by the existing rules, which would greatly accelerate the process of finding gaps in the training data during the initial spike of the learning curve).
       4. Besides the improved explainability, with this approach the authors were able to match BERT's performance in a relation extraction task using between 2x and 20x less training data, and outperform it by 3%-10% F1 when using the full dataset.



