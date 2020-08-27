import numpy as np
import random
import sys

from collections import Counter, defaultdict as deft

from copy import deepcopy as cp

from Dataset import TitleDataset, TitleTextDataset
 
from FeatureExtraction import STOPWORDS

from nltk import (
    ngrams,
    sent_tokenize as splitter,
    wordpunct_tokenize as tokenizer
)

from scipy.spatial.distance import cosine

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

from tools import (
    to_json
)

from tqdm import tqdm




def title(X, docid):
    return splitter(X[docid])[0]
    return '%s...' % X[docid][:90]


class Cluster:

    def __init__(self, X, vec, mat, title, docid, sims):
        self._X = X
        self._vec = vec
        self._mat = mat
        self._features = self._vec.get_feature_names()
        self.id = docid
        self.titles = {docid: title}
        self.cluster_name = None
        if sims:
            sims, ids, titles = list(zip(*sims))
            self.elements = set(ids)
            self.titles.update(dict(zip(ids, titles)))
        else:
            self.elements = set([docid])
    
    def items(self):
        docids = cp(self.elements)
        docids.add(self.id)
        return sorted(list(docids))
    
    def dump(self):
        for docid in self.items():
            yield self.__make_record(docid)
    
    def __get_tfidf_weight(self, token):
        w = 0
        try:
            dimid = self._features.index(token)
        except ValueError:
            return w
        for docid in self.items():
            w += self._mat[docid][dimid]
        return w
    
    def make_cluster_name(self):
        if self.cluster_name:
            return self.cluster_name
        all_titles = [
            [w.lower() for w in tokenizer(self.titles[docid])]
            for docid in self.items()
        ]
        gram_dist = Counter()
        for title_tokens in random.sample(
            all_titles, 50 if 50 < len(all_titles) else len(all_titles)
        ):
            for n in [1, 2, 3, 4, 5]:
                for gram in ngrams(title_tokens, n):
                    tfidf_weight = sum([
                        self.__get_tfidf_weight(token.lower())
                        for token in gram
                    ])
                    if (
                        gram[0].lower() in STOPWORDS
                        or gram[-1].lower() in STOPWORDS
                        or [token for token in gram if not token.isalpha()]
                    ):
                        continue
                    gram_dist[gram] += tfidf_weight * n
        if not gram_dist.most_common():
            return None
        self.cluster_name = ' '.join(gram_dist.most_common()[0][0])
        return self.cluster_name
        
    
    def __make_record(self, docid):
        return {
            'id': None,
            'lang': 'en',
            'date': None,
            'title': self.titles[docid],
            'cluster_name': self.make_cluster_name(),
            'cluster': self.id,
            'text': self._X[docid].replace(self.titles[docid], '')
        }
    
    def __str__(self):
        return '<%d %d>' % (self.id, len(self))

    def __contains__(self, cluster, max_overlap=0.34):
        a = self.elements
        b = cluster.elements
        shared = a.intersection(b)
        if len(shared) / len(b) >= max_overlap:
            return True
        return False
    
    def __len__(self):
        return len(self.items())



def make_inverted_index(mat):
    docids_by_dimid = deft(set)
    for docid, v in enumerate(mat):
        for dimid, w in enumerate(v):
            if w:
                docids_by_dimid[dimid].add(docid)
    return docids_by_dimid


def lookup_on_index(mat, docids_by_dimid, docid, n=100, radius=0.3):
    docids = Counter()
    for dimid, w in enumerate(mat[docid]):
        if w:
            docids.update(docids_by_dimid[dimid])
    return [
        _docid for _docid, freq in docids.most_common(n)
        if cosine(mat[docid], mat[_docid]) <= radius
    ]


def calculate_relevance_threshold(
    lower_bound_ratio,
    sims_by_docid,
    coef=0.3
):
    top_sims = [
        min([record[0] for record in sims])
        for sims in sims_by_docid.values()
        if sims
    ]
    least_top_sims_n = int(len(top_sims) * lower_bound_ratio)
    least_top_sims_n = least_top_sims_n if least_top_sims_n > 3 else 3
    least_top_sims = sorted(top_sims)[:least_top_sims_n]
    print(least_top_sims)
    relevance_threshold = (sum(least_top_sims) / least_top_sims_n) * coef
    print(top_sims)
    print(relevance_threshold)
    print('---')
    return relevance_threshold


def calculate_elementwise_similarities(_mat):
    docids_by_dimid = make_inverted_index(_mat)
    sims_by_docid = dict([])
    docids = set(range(len(_mat)))
    vals = []
    for docid in tqdm(docids):
        sims = []
        v = mat[docid]
        for _docid in lookup_on_index(_mat, docids_by_dimid, docid):
            if docid == _docid:
                continue
            _v = mat[_docid]
            sim = 1 - cosine(v, _v)
            record = (sim, _docid, title(X, _docid))
            sims.append(record)
            vals.append(sim)
        sims.sort(key=lambda x: x[0], reverse=True)
        sims_by_docid[docid] = sims
    return sims_by_docid


def clusterize(
    X, vec, mat, lower_bound_ratio=0.05, ratio_over_best=0.6
):

    _mat = mat.tolist()    
    sims_by_docid = calculate_elementwise_similarities(_mat)
        
    relevance_threshold = calculate_relevance_threshold(
        lower_bound_ratio, sims_by_docid
    )

    clusters = []
    space = list(sims_by_docid.items())
    for docid, sims in tqdm(space):

        if sims:        
            best_sim = sims[0][0]
            sims = [
                record for record in sims
                if record[0] / best_sim >= ratio_over_best
                and record[0] >= relevance_threshold
            ]
        
        cl = Cluster(X, vec, _mat, title(X, docid), docid, sims)

        if not clusters:
            clusters.append(cl)
            continue

        clusterized = False
        for _cl in clusters:
            if cl in _cl:
                clusterized = True
                break
        if not clusterized:
            clusters.append(cl)

    return clusters



if __name__ == '__main__':

    d = TitleTextDataset()
    
    X, Y = list(zip(*d))
    
    #X = random.sample(X, int(len(X) * 0.2))
    
    vec = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.5)
    mat = vec.fit_transform(X).todense()

    print('Clusterizing...')
    clusters = clusterize(X, vec, mat)
    
    payload = []
    print('Dumping...')
    for cl in tqdm(sorted(clusters, key=lambda x: len(x), reverse=True)):
        for record in cl.dump():
            payload.append(record)
    to_json(payload, 'clusters.out.json', indent=4)
    


    