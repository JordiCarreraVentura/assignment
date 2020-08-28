from sklearn.model_selection import GridSearchCV

from tools import (
    from_json,
    from_pickle
)


class DeserializedModelWrapper:

    def __init__(self, payload):
        self._payload = payload
        print(type(self._payload))
        if isinstance(self._payload, GridSearchCV):
            self._predictor = self._payload.best_estimator_.predict
        else:
            raise NotImplementedError
    
    def __call__(self, data):
        if isinstance(data, list):
            return self._predictor(data)
        else:
            return self._predictor([data])[0]



if __name__ == '__main__':
    
    model_path = 'models/ngrams.TitleTextDataset.svm_ngrams.p'
    model_path = 'models/sentence_transformer__sentence_encoders.p'
    
    model = DeserializedModelWrapper(from_pickle(model_path))
    
    X = [
        '%s. %s' % (o['title'], o['text'])
        for o in from_json('data/fake_test.json')
    ]
    
    Y = model(X)
    
    for x, y in zip(X, Y):
        print('%s...' % x[:250])
        print('predicted=%d expected=%d\n' % (y, 1))
    
    
    
    