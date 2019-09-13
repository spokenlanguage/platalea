# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
import numpy
from scipy.spatial.distance import cdist

def cosine(x, y):
    return cdist(x, y, metric='cosine')

def ranking(candidates, references, correct, metric=cosine, ns=(1, 5, 10)):
    """Rank `candidates` in order of similarity for each vector and return evaluation metrics.

    `correct[i][j]` indicates whether for reference item i the candidate j is correct.
    """
    distances = cdist(references, candidates)
    result = {'ranks' : [] , 'recall' : {} }
    for n in ns:
        result['recall'][n]    = []
    for j, row in enumerate(distances):
        ranked = numpy.argsort(row)
        id_correct = numpy.where(correct[j][ranked])[0]
        rank1 = id_correct[0] + 1
        for n in ns:
            id_topn = ranked[:n]
            overlap = len(set(id_topn).intersection(set(ranked[id_correct])))
            result['recall'][n].append(overlap/len(id_correct))
        result['ranks'].append(rank1)
    return result
