from __future__ import print_function, division

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


if __name__ == '__main__':    
    
    # Hypothesis 1: input clean-up improves performance (refuted)
    hypothesis, datasets, pipelines = HYPOTHESIS_1
    
    # Hypothesis 2: title information improves performance (confirmed, +~1% F1)
    hypothesis, datasets, pipelines = HYPOTHESIS_2
    
    # Hypothesis 3: n-gram features improve performance
    # (confirmed for discriminative models, ~+1% F1;
    # refuted for Naive Bayes)
    hypothesis, datasets, pipelines = HYPOTHESIS_3
    
    # Hypothesis 4: character n-gram features improve performance
    # (refuted, -0.5-1% F1 across all models; there may be some
    # generalization benefits thanks to the use of subword features
    # but the effect does not seem noticeable in data from the same
    # domain)
    hypothesis, datasets, pipelines = HYPOTHESIS_4

    # Hypothesis 5: standard exploration of the parameter space 
    # using grid search
    hypothesis, datasets, pipelines = HYPOTHESIS_5
    

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

            print(grid.best_estimator_.named_steps.items())

            # Predict on the held-out split
            vec = grid.best_estimator_['vec']
            cls = grid.best_estimator_['cls']
            X, Y = list(zip(*d.test()))
            Y_ = cls.predict(vec.transform(X))

            args = (di + 1, len(datasets), pj + 1, len(pipelines), ppln_name)
            print('\n\n(%d/%d, %d/%d)\n========== %s ==========' % args)
            print(grid.best_score_)
            print('---------- grid-search ----------\n', classification_report(Y, Y_))
        
            rows = [[] for _ in range(len(list(grid.cv_results_.values())[0]))]
            keys = []
            for key, col_array in grid.cv_results_.items():
                keys.append(key)
                for i, cell in enumerate(col_array):
                    rows[i].append(cell)
            _rows = [keys] + sorted(
                rows,
                reverse=True,
                key=lambda x: x[keys.index('mean_test_score')]
            )
            csv_name = '.'.join([str(d), ppln_name])
            to_csv(_rows, 'reports/%s.%s.csv' % (hypothesis, csv_name))
            to_txt(
                str(classification_report(Y, Y_)),
                'reports/%s.%s.txt' % (hypothesis, csv_name)
            )
            to_pickle(grid, 'models/%s.%s.p' % (hypothesis, csv_name))
