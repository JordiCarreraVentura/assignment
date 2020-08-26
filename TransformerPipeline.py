import numpy as np
import sys

from Dataset import (
    NormalizedTitleTextDataset,
    TextDataset,
    TitleDataset,
    TitleTextDataset
)
 
from hypotheses import (
    HYPOTHESIS_6,
    HYPOTHESIS_7
)

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

# from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from tools import (
    to_csv,
    to_pickle,
)



def summarize_csv(rows, columns):
    keys = list(rows[0])
    rows = rows[1:]
    out = [columns]
    for row in rows:
        _row = [row[keys.index(col)] for col in columns]
        out.append(_row)
    return out


def report(k, hypothesis, classifier, rows, grid):
    setting_name = '%s.%s' % (hypothesis, classifier.__class__.__name__)
    _rows = [[] for _ in range(len(list(grid.cv_results_.values())[0]))]
    k_row = [k for _ in range(len(list(grid.cv_results_.values())[0]))]
    for key, col_array in ([('k', k_row)] + list(grid.cv_results_.items())):
        if len(keys) < len(grid.cv_results_.keys()) + 1:
            keys.append(key)
        for i, cell in enumerate(col_array):
            _rows[i].append(cell)
    rows += _rows

    _rows = sorted(
        _rows,
        reverse=True,
        key=lambda x: x[keys.index('mean_test_score')]
    )
    to_csv(
        [keys] + _rows,
        'reports/%s.csv' % setting_name
    )
    to_csv(
        summarize_csv([keys] + _rows, columns),
        'reports/%s.summary.csv' % setting_name
    )

    to_pickle(grid, 'models/%s.p' % setting_name)



CV = 3
N_JOBS = 1
VERBOSITY = 1

HYPOTHESES = {
    '6': HYPOTHESIS_6,
    '7': HYPOTHESIS_7
}


if __name__ == '__main__':

    model = SentenceTransformer('distiluse-base-multilingual-cased')
    
    dataset = TitleTextDataset()
    
    K = [15, 100, len(dataset)]
    #K = [15]
    
    for k in K:
        
        train_X, Y_train = list(zip(*dataset.sample(k)))
        test_X, Y_test = list(zip(*dataset.test()))
        #test_X, Y_test = list(zip(*dataset.sample(5, test=True)))

        skf = StratifiedKFold(n_splits=CV)
        folds = list(skf.split(train_X, Y_train))

        hypothesis_id = sys.argv[1]
        hypothesis, classifiers, param_grid, columns = HYPOTHESES[hypothesis_id]
    
        rows = []
        keys = []
        for classifier in classifiers:

            arg_val = 0

            _param_grid = param_grid
            _param_grid['encoder__model'] = [model]


            ppln = Pipeline([
                ('encoder', None),
                ('cls', classifier)
            ])

            # 'GridSearchCV' defaults to stratified k-fold
            grid = GridSearchCV(
                ppln, cv=folds, n_jobs=N_JOBS, verbose=VERBOSITY,
                param_grid=_param_grid,
                refit=True
            )

            grid.fit(train_X, Y_train)

            Y_pred = grid.predict(test_X)

            print(grid.best_estimator_.named_steps)
        
            report(k, hypothesis, classifier, rows, grid)
