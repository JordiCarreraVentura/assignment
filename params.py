from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


param_grids = {

    'nb_baseline': [
        {
            'vec': [CountVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'svm_baseline': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'log_baseline': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],


    # Hypothesis 1: data clean-up improves performance
    'nb_fext': [
        {
            'fext__remove_entities': [True, False],
            'fext__remove_nonalpha': [True, False],
            'fext__remove_stopwords': [True, False],
            'vec': [CountVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'svm_fext': [
        {
            'fext__remove_entities': [True, False],
            'fext__remove_nonalpha': [True, False],
            'fext__remove_stopwords': [True, False],
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'log_fext': [
        {
            'fext__remove_entities': [True, False],
            'fext__remove_nonalpha': [True, False],
            'fext__remove_stopwords': [True, False],
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],


    # Hypothesis 2: data clean-up improves performance
    # (This hypothesis does not use specific parameters).


    # Hypothesis 3: n-gram features improve performance
    'nb_ngrams': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            'vec__max_features': [15000, 20000, 25000],
            'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'svm_ngrams': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000, 20000, 25000],
            'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],
    
    'log_ngrams': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [15000, 20000, 25000],
            'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
        }
    ],


    # Hypothesis 4: character n-gram features improve performance
    'nb_char': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            'vec__max_df': [0.2, 0.5, 0.7],
            'vec__min_df': [5, 10, 100],
        }
    ],
    
    'svm_char': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            'vec__max_df': [0.2, 0.5, 0.7],
            'vec__min_df': [5, 10, 100],
        }
    ],
    
    'log_char': [
        {
            'vec': [TfidfVectorizer()],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            'vec__max_df': [0.2, 0.5, 0.7],
            'vec__min_df': [5, 10, 100],
        }
    ],









    # Other hypotheses

    'nb_word': [
        {
#             'fext__remove_entities': [True, False],
#             'fext__remove_nonalpha': [True, False],
#             'fext__remove_stopwords': [True, False],
            'vec': [CountVectorizer(), TfidfVectorizer()],
            'vec__max_features': [15000],
            'vec__ngram_range': [(1, 1)],
            'vec__analyzer': ['word'],
            'vec__max_df': [0.5],
            'vec__min_df': [2],
            
#             'vec': [CountVectorizer(), TfidfVectorizer()],
#             'vec__max_features': [10000, 20000, 30000, 1000000],
#             'vec__ngram_range': [(1, 1), (1, 2)],
#             'vec__analyzer': ['word'],
#             'vec__max_df': [0.2, 0.5, 0.8],
#             'vec__min_df': [2, 5, 10],

#             'vec': [CountVectorizer(), TfidfVectorizer()],
#             'vec__max_features': [10000, 20000],
#             'vec__ngram_range': [(1, 1), (1, 2)],
#             'vec__analyzer': ['word'],
#             'vec__max_df': [0.2, 0.5],
#             'vec__min_df': [2, 5],
        }
    ],
    
    'svm_word': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [10000, 20000],
            'vec__ngram_range': [(1, 1), (1, 2)],
            'vec__analyzer': ['word'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
        }
    ],

    'nb_char': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                #(2, 3), (3, 4), (4, 5), (2, 4), (2, 5), (3, 5)
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
        }
    ],
    
    'svm_char': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                #(2, 3), (3, 4), (4, 5), (2, 4), (2, 5), (3, 5)
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
        }
    ],
    
    'log_word': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [10000, 20000],
            'vec__ngram_range': [(1, 1), (1, 2)],
            'vec__analyzer': ['word'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
            'cls__solver': ['lbfgs'],
            #'cls__tol': [0.0001, 0.001, 0.005, 0.01],
            #'cls__C': [0.5, 0.8, 1.0],
            'cls__tol': [0.0001],
            'cls__C': [1.0],
            #'cls__max_iter': [50, 100, 200, 500],
            #'cls__max_iter': [50, 100, 200],
            'cls__max_iter': [100],
            'cls__penalty': ['l2'],
            #'cls__penalty': ['l1', 'l2'],
        },
    ],
    
    'log_char': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [2000, 5000],
            'vec__ngram_range': [
                #(2, 3), (3, 4), (4, 5), (2, 4), (2, 5), (3, 5)
                (2, 3), (2, 5), (3, 5)
            ],
            'vec__analyzer': ['char'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
            'cls__solver': ['lbfgs'],
            #'cls__tol': [0.0001, 0.001, 0.005, 0.01],
            #'cls__C': [0.5, 0.8, 1.0],
            'cls__tol': [0.0001],
            'cls__C': [1.0],
            #'cls__max_iter': [50, 100, 200, 500],
            #'cls__max_iter': [50, 100, 200],
            'cls__max_iter': [100],
            'cls__penalty': ['l2'],
            #'cls__penalty': ['l1', 'l2'],
        }
    ],
    
    'dt_word': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [10000, 20000],
            'vec__ngram_range': [(1, 1), (1, 2)],
            'vec__analyzer': ['word'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
            'cls__n_estimators': [5, 10, 20],
            'cls__min_samples_split': [2, 5],
            'cls__min_samples_leaf': [3, 5]   
        }
    ],
    
    'gb_word': [
        {
            'vec': [CountVectorizer(), TfidfVectorizer()],
            #'vec__max_features': [10000, 20000, 30000, 1000000],
            'vec__max_features': [10000, 20000],
            'vec__ngram_range': [(1, 1), (1, 2)],
            'vec__analyzer': ['word'],
            #'vec__max_df': [0.2, 0.5, 0.8],
            'vec__max_df': [0.2, 0.5],
            #'vec__min_df': [2, 5, 10],
            'vec__min_df': [2, 5],
            'cls__n_estimators': [5, 10, 20],
            'cls__learning_rate': [0.1, 0.25, 0.5],
            'cls__min_samples_split': [2, 10],
            'cls__max_depth': [3, 5],
            'cls__tol': [0.0001, 0.001, 0.01],
            'cls__validation_fraction': [0.1, 0.2, 0.3],  
        }
    ],
    
}