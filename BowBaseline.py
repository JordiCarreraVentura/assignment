from __future__ import print_function, division

import numpy as np

from Dataset import TitleTextDataset


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



if __name__ == '__main__':

    d = TitleTextDataset()

    ppln = Pipeline([
        ('vec', TfidfVectorizer()),
        #('cls', LinearSVC())
        ('cls', LogisticRegression())
    ])

    param_grid = [
        {
            'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__ngram_range': [(1, 1), (1, 2)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.2, 0.5, 0.8],
            'vec__min_df': [2, 5, 10],
            'cls__solver': ['lbfgs'],
            'cls__tol': [0.0001, 0.001, 0.005, 0.01],
            'cls__C': [0.5, 0.8, 1.0],
            'cls__max_iter': [50, 100, 200, 500],
            'cls__penalty': ['l2'],
            #'cls__penalty': ['l1', 'l2'],
        },
        {
            'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__ngram_range': [
                (2, 3), (3, 4), (4, 5), (2, 4), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            'vec__max_df': [0.2, 0.5, 0.8],
            'vec__min_df': [2, 5, 10],
            'cls__solver': ['lbfgs'],
            'cls__tol': [0.0001, 0.001, 0.005, 0.01],
            'cls__C': [0.5, 0.8, 1.0],
            'cls__max_iter': [50, 100, 200, 500],
            'cls__penalty': ['l2'],
            #'cls__penalty': ['l1', 'l2'],
        }
    ]

    # 'GridSearchCV' defaults to stratified k-fold
    grid = GridSearchCV(ppln, cv=5, n_jobs=4, param_grid=param_grid, verbose=4)

    X, Y = list(zip(*d))
    grid.fit(X, Y)

    print(grid.best_estimator_.named_steps['cls'])
