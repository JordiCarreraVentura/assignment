from __future__ import print_function, division

import sys
import numpy as np

from Dataset import (
    NormalizedTitleTextDataset,
    TextDataset,
    TitleDataset,
    TitleTextDataset
)

from FeatureExtraction import FeatureExtractor

from hypotheses import (
    HYPOTHESIS_1,
    HYPOTHESIS_2,
    HYPOTHESIS_3,
    HYPOTHESIS_4,
    HYPOTHESIS_5
)

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
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from tools import (
    make_dirs as mkdir,
    to_csv,
    to_pickle,
    to_txt
)



CV = 3
N_JOBS = 8
VERBOSITY = 0

HYPOTHESES = {

    # Hypothesis 1: input clean-up improves performance (refuted)
    '1': HYPOTHESIS_1,

    # Hypothesis 2: title information improves performance (confirmed, +~1% F1)
    '2': HYPOTHESIS_2,
    
    # Hypothesis 3: n-gram features improve performance
    # (confirmed for discriminative models, ~+1% F1;
    # refuted for Naive Bayes)
    '3': HYPOTHESIS_3,
    
    # Hypothesis 4: character n-gram features improve performance
    # (refuted, -0.5-1% F1 across all models; there may be some
    # generalization benefits thanks to the use of subword features
    # but the effect does not seem noticeable in data from the same
    # domain)
    '4': HYPOTHESIS_4,

    # Hypothesis 5: standard exploration of the parameter space 
    # using grid search
    '5': HYPOTHESIS_5
}


def precision(Y, _Y):
    return round(precision_score(Y, _Y), 2)

def recall(Y, _Y):
    return round(recall_score(Y, _Y), 2)

def f1(Y, _Y):
    return round(f1_score(Y, _Y), 2)

def accuracy(Y, _Y):
    return round(accuracy_score(Y, _Y), 2)



def report(d, hypothesis, ppln_name, grid):

    row_range = list(range(len(list(grid.cv_results_.values())[0])))
    rows = [[] for _ in row_range]
    keys = []
    csv_name = '.'.join([hypothesis, d.__class__.__name__, ppln_name])
    for key, col_array in (list(grid.cv_results_.items())):
        if len(keys) < len(grid.cv_results_.keys()):
            keys.append(key)
        for i, cell in enumerate(col_array):
            rows[i].append(cell)

    rows = sorted(
        rows,
        reverse=True,
        key=lambda x: x[keys.index('mean_test_score')]
    )

    to_pickle(grid, 'models/%s.p' % csv_name)
    
    to_csv(
        [keys] + rows,
        'reports/%s.csv' % csv_name
    )




if __name__ == '__main__':    
    
    hypothesis_id = sys.argv[1]
    
    hypothesis, datasets, pipelines = HYPOTHESES[hypothesis_id]

    rows = [(
        'dataset', 'pipeline', 'vectorizer', 'classifier',
        'precision', 'recall', 'f1', 'accuracy'
    )]
    for di, dataset in enumerate(datasets):
        for pj, (ppln_name, ppln) in enumerate(pipelines):

            d = dataset()
            X, Y = list(zip(*d.train()))
        
            param_grid = param_grids[ppln_name]

            # 'GridSearchCV' defaults to stratified k-fold
            grid = GridSearchCV(
                ppln, cv=CV, n_jobs=N_JOBS, verbose=VERBOSITY,
                param_grid=param_grid,
                refit=True
            )
            grid.fit(X, Y)

            #print(grid.best_estimator_.named_steps.items())
        
            report(d, hypothesis, ppln_name, grid)
            

            # Predict on the held-out split
            vec = grid.best_estimator_['vec']
            cls = grid.best_estimator_['cls']
            X, Y = list(zip(*d.test()))
            Y_ = cls.predict(vec.transform(X))
            
            p = precision(Y, Y_)
            r = recall(Y, Y_)
            f = f1(Y, Y_)
            a = accuracy(Y, Y_)
            row = (
                d.__class__.__name__, ppln_name,
                vec.__class__.__name__,
                cls.__class__.__name__, 
                p, r, f, a
            )
            rows.append(row)
            to_csv(rows, 'reports/%s.csv' % hypothesis)
