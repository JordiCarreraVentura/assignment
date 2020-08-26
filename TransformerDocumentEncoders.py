import numpy as np
import os

from nltk import sent_tokenize as splitter

from sklearn.base import BaseEstimator

from tools import (
    from_pickle,
    to_pickle
)


# https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-np
def pooling(mat, ksize, method='max', pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method == 'max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    elif method == 'avg':
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))
    elif method == 'mdn':
        result=np.median(mat_pad.reshape(new_shape),axis=(1,3))

    return result



class Encoder(BaseEstimator):

    def __init__(
        self,
        model=None
    ):
        self.model = model
        self.cache = dict([])

    def fit(self, X, Y):
        return self
    
    def __str__(self):
        return self.__class__.__name__
    
    def to_cache(self, X, _X):
        arg = '_'.join([x[:90] for x in X])
        self.cache[arg] = _X
    
    def from_cache(self, X):
        arg = '_'.join([x[:90] for x in X])
        if arg in self.cache:
            return (True, self.cache[arg])
        else:
            return (False, None)       

#     def pickle_name(self):
#         return 'bin/%s.p' % str(self)
#     
#     def to_cache(self, _X):
#         to_pickle(_X, self.pickle_name())
#     
#     def from_cache(self):
#         name = self.pickle_name()
#         if os.path.exists(name):
#             return True, from_pickle(name)
#         return False, None
    
    

class StringConcatenationEncoder(Encoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = self.model.encode(X)
            self.to_cache(X, _X)
        if Y:
            return _X, Y
        else:
            return _X



class SentenceLevelAveragePooler(Encoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                encoded_sents = self.model.encode(sents)
                doc_vecs = pooling(
                    encoded_sents,
                    (encoded_sents.shape[0], 1),
                    method='avg'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)



class SentenceLevelMaximumPooler(Encoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                encoded_sents = self.model.encode(sents)
                doc_vecs = pooling(
                    encoded_sents,
                    (encoded_sents.shape[0], 1),
                    method='max'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)



class SentenceLevelMedianPooler(Encoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                encoded_sents = self.model.encode(sents)
                doc_vecs = pooling(
                    encoded_sents,
                    (encoded_sents.shape[0], 1),
                    method='mdn'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)




class BlockEncoder(Encoder):
    
    def __init__(self, model=None, block_size=3):
        super().__init__(model)
        self.block_size = block_size
    
    def __str__(self):
        return self.__class__.__name__



class SentenceBlockMaximumPooler(BlockEncoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                blocks, block = [], []
                while sents:
                    sent = sents.pop(0)
                    block.append(sent)
                    if len(block) == self.block_size:
                        blocks.append(' '.join(block))
                        block = []
                blocks.append(' '.join(block))
                encoded_blocks = self.model.encode(blocks)
                doc_vecs = pooling(
                    encoded_blocks,
                    (encoded_blocks.shape[0], 1),
                    method='max'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)




class SentenceBlockMedianPooler(BlockEncoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                blocks, block = [], []
                while sents:
                    sent = sents.pop(0)
                    block.append(sent)
                    if len(block) == self.block_size:
                        blocks.append(' '.join(block))
                        block = []
                blocks.append(' '.join(block))
                encoded_blocks = self.model.encode(blocks)
                doc_vecs = pooling(
                    encoded_blocks,
                    (encoded_blocks.shape[0], 1),
                    method='mdn'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)


class SentenceBlockAveragePooler(BlockEncoder):
    
    def transform(self, X, Y=None):
        success, _X = self.from_cache(X)
        if not success:
            _X = []
            for document in X:
                sents = splitter(document)
                blocks, block = [], []
                while sents:
                    sent = sents.pop(0)
                    block.append(sent)
                    if len(block) == self.block_size:
                        blocks.append(' '.join(block))
                        block = []
                blocks.append(' '.join(block))
                encoded_blocks = self.model.encode(blocks)
                doc_vecs = pooling(
                    encoded_blocks,
                    (encoded_blocks.shape[0], 1),
                    method='avg'
                )
                _X.append(list(doc_vecs)[0])
            self.to_cache(X, _X)
        if Y:
            return np.array(_X), Y
        else:
            return np.array(_X)