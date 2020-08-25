import nltk

from nltk import (
    sent_tokenize as splitter,
    wordpunct_tokenize as tokenizer
)

from nltk.corpus import stopwords

from sklearn.base import BaseEstimator

STOPWORDS = stopwords.words('english')



class FeatureExtractor(BaseEstimator):

    def __init__(
        self,
        remove_entities=False,
        remove_nonalpha=False,
        remove_stopwords=False,
        lowercase=False
    ):
        self.remove_entities = remove_entities
        self.remove_nonalpha = remove_nonalpha
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
    
    def fit(self, X, Y):
        return self
    
    def transform(self, X, Y=None):
        _X = []
        for x in X:
            _X.append(self(x))
        if Y:
            return _X, Y
        else:
            return _X


    def __call__(self, x):
        bow = []
        for sent in splitter(x):
            for i, token in enumerate(tokenizer(sent)):

                if self.remove_nonalpha and not token.isalpha():
                    continue
                
                if (
                    self.remove_entities
                    and i
                    and token[0] != token[0].lower()
                ):
                    continue
                
                if (
                    self.remove_stopwords
                    and token.lower() in STOPWORDS
                ):
                    continue
            
                bow.append(token if not self.lowercase else token.lower())
        
        _bow = []
        prev = None
        while bow:
            token = bow.pop(0)
            if token == prev:
                continue
            _bow.append(token)
            prev = token
        
        return ' '.join(_bow)
