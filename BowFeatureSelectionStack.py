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
    HYPOTHESIS_8
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

    # Hypothesis 8: model stacking improves performance
    '8': HYPOTHESIS_8
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

    keys = (
        'dataset', 'pipeline', 'vectorizer', 'feature extractor',
        'classifier', 'precision', 'recall', 'f1', 'accuracy'
    )
    arg_val = 0
    rows = []
    for stack in [
        LinearSVC,
        LogisticRegression,
        MultinomialNB,
        RandomForestClassifier
    ]:
    
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

                report(d, hypothesis, ppln_name, grid)

            
                # We will use the top-performing parameters of the BoW-TFIDF 
                # pipeline as the parameters for the dimensionality reduction.
                # 
                # We will use those parameters to regenerate the X matrix (which
                # has remained implicit as the vectorization step inside the
                # `sklearn.pipeline.Pipeline`).
                #
                # Then, we will apply the feature reduction tranformation on
                # that matrix to obtain the new matrix and train a new model,
                # reproducing manually the boosting logic used in e.g. gradient
                # boosting classifiers:
                vec = grid.best_estimator_['vec']
                cls = grid.best_estimator_['cls']
                _X = vec.transform(X)
                reducer = SelectFromModel(cls, prefit=True, max_features=15000)
                _Xr = reducer.transform(_X)

                # Train stacked classifier
                stacked = stack()
                stacked.fit(_Xr, Y)

                # Predict with stacked classifier
                __X, __Y = list(zip(*d.test()))
                Y_ = stacked.predict(reducer.transform(vec.transform(__X)))
            
                p = precision(__Y, Y_)
                r = recall(__Y, Y_)
                f = f1(__Y, Y_)
                a = accuracy(__Y, Y_)
                row = (
                    d.__class__.__name__, ppln_name,
                    vec.__class__.__name__,
                    cls.__class__.__name__,
                    stacked.__class__.__name__,
                    p, r, f, a
                )
                rows.append(row)
                to_csv(
                    [keys] + sorted(rows, reverse=True, key=lambda x: x[-2]),
                    'reports/%s.csv' % hypothesis
                )
                
                if f > arg_val:
                    arg_val = f
                    model = {
                        'grid': grid,
                        'reducer': reducer,
                        'stacked': stacked
                    }
                    to_pickle(model, 'models/%s.p' % hypothesis)



