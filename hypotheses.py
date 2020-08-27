from Dataset import (
    ClustersDataset,
    NormalizedTitleTextDataset,
    TextDataset,
    TitleDataset,
    TitleTextDataset
)

from FeatureExtraction import FeatureExtractor

from params import param_grids

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
    
from TransformerDocumentEncoders import (
    SentenceBlockAveragePooler,
    SentenceBlockMaximumPooler,
    SentenceBlockMedianPooler,
    SentenceLevelAveragePooler,
    SentenceLevelMedianPooler,
    SentenceLevelMaximumPooler,
    StringConcatenationEncoder
)




    
# Hypothesis 1: input clean-up improves performance

datasets = [TitleTextDataset]
hypothesis = 'fext'


pipelines = [

    ('nb_baseline', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_baseline', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_baseline', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ])),

    ('nb_fext', Pipeline([
        ('fext', FeatureExtractor()),
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_fext', Pipeline([
        ('fext', FeatureExtractor()),
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_fext', Pipeline([
        ('fext', FeatureExtractor()),
        ('vec', None),
        ('cls', LogisticRegression())
    ]))
]

HYPOTHESIS_1 = (hypothesis, datasets, pipelines)




# Hypothesis 2: title information improves performance

datasets = [TextDataset, TitleDataset, TitleTextDataset]
hypothesis = 'title'
pipelines = [

    ('nb_baseline', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_baseline', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_baseline', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ]))
]

HYPOTHESIS_2 = (hypothesis, datasets, pipelines)




# Hypothesis 3: n-gram features improve performance

datasets = [TitleTextDataset]
hypothesis = 'ngrams'
pipelines = [

    ('nb_ngrams', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_ngrams', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_ngrams', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ]))
]

HYPOTHESIS_3 = (hypothesis, datasets, pipelines)




# Hypothesis 4: character n-gram features improve performance

datasets = [TitleTextDataset]
hypothesis = 'char'
pipelines = [

    ('nb_char', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_char', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_char', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ]))
]

HYPOTHESIS_4 = (hypothesis, datasets, pipelines)




# Hypothesis 5: standard parameter exploration

datasets = [TitleTextDataset]
hypothesis = 'param'
pipelines = [

    ('nb_param', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_param', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_param', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ])),
    
    ('dt_param', Pipeline([
        ('vec', None),
        ('cls', RandomForestClassifier()),
    ])),

    ('gb_param', Pipeline([
        ('vec', None),
        ('cls', GradientBoostingClassifier()),
    ])),

]

HYPOTHESIS_5 = (hypothesis, datasets, pipelines)






# Hypothesis 6: block size impacts performance

hypothesis = 'sentence_transformer__block_size'

classifiers = [
    LogisticRegression(),
    LinearSVC(),
    RandomForestClassifier()
]

columns = 'k,classifier,mean_test_score,std_test_score,rank_test_score,param_encoder,param_encoder__block_size'.split(',')

param_grid = {
    'encoder': [
        SentenceBlockAveragePooler(),
        SentenceBlockMaximumPooler(),
        SentenceBlockMedianPooler()
    ],
    'encoder__block_size': [1, 3, 5, 10]
    #'encoder__block_size': [3]
}

HYPOTHESIS_6 = (hypothesis, classifiers, param_grid, columns)




# Hypothesis 7: PENDING

hypothesis = 'sentence_transformer__sentence_encoders'

columns = 'k,classifier,mean_test_score,std_test_score,rank_test_score,param_encoder'.split(',')

classifiers = [
    LogisticRegression(),
    LinearSVC(),
    RandomForestClassifier()
]

param_grid = {
    'encoder': [
        SentenceLevelAveragePooler(),
        SentenceLevelMaximumPooler(),
        SentenceLevelMedianPooler(),
        StringConcatenationEncoder(),
    ]
}

HYPOTHESIS_7 = (hypothesis, classifiers, param_grid, columns)




# Hypothesis 8: model stacking improves performance

datasets = [TitleTextDataset]
hypothesis = 'stacking'
pipelines = [

    ('nb_stack', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_stack', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ])),

    ('log_stack', Pipeline([
        ('vec', None),
        ('cls', LogisticRegression())
    ])),

    ('dt_stack', Pipeline([
        ('vec', None),
        ('cls', RandomForestClassifier())
    ]))
]

HYPOTHESIS_8 = (hypothesis, datasets, pipelines)







# Hypothesis 9: clustering is a BoW problem

datasets = [ClustersDataset]
hypothesis = 'clusters'
pipelines = [

    ('nb_clusters', Pipeline([
        ('vec', None),
        ('cls', MultinomialNB())
    ])),

    ('svm_clusters', Pipeline([
        ('vec', None),
        ('cls', LinearSVC())
    ]))

]

HYPOTHESIS_9 = (hypothesis, datasets, pipelines)