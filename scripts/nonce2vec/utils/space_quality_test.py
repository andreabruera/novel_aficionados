#Functions for calculating the MEN score for word embeddings created with Gensim W2V
import argparse
import pickle
import os
import gensim
import numpy as np
import collections
import nonce2vec
import scipy

from collections import defaultdict

from scipy import stats, sparse

from gensim.models import Word2Vec
from nonce2vec.utils.count_based_models_utils import *

#def cosine_similarity(vector_1, vector_2): 
    #cosine_similarity = dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
    #return cosine_similarity

def score(gold, prediction, method):
    if len(gold) != len(prediction):
        raise ValueError("The two arrays must have the same length!")
    gold = np.array(gold, dtype=np.double)
    prediction = np.array(prediction, dtype=np.double)
    if method == "pearson":
        return pearson(gold, prediction)[0]
    elif method == "spearman":
        return spearman(gold, prediction)[0]

def pearson(gold, prediction):
    return stats.pearsonr(gold, prediction)

def spearman(gold, prediction):
    return stats.spearmanr(gold, prediction, None)

def ppmi(csr_matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logging.info('Weighing raw count CSR matrix via PPMI')
    words = sparse.csr_matrix(csr_matrix.sum(axis=1))
    contexts = sparse.csr_matrix(csr_matrix.sum(axis=0))
    total_sum = csr_matrix.sum()
    # csr_matrix = csr_matrix.multiply(words.power(-1)) # #(w, c) / #w
    # csr_matrix = csr_matrix.multiply(contexts.power(-1))  # #(w, c) / (#w * #c)
    # csr_matrix = csr_matrix.multiply(total)  # #(w, c) * D / (#w * #c)
    csr_matrix = csr_matrix.multiply(words.power(-1, dtype=float))\
                           .multiply(contexts.power(-1, dtype=float))\
                           .multiply(total_sum)
    csr_matrix.data = np.log2(csr_matrix.data)  # PMI = log(#(w, c) * D / (#w * #c))
    csr_matrix = csr_matrix.multiply(csr_matrix > 0)  # PPMI
    csr_matrix.eliminate_zeros()
    return csr_matrix

def sim_check(training_mode, model, vocabulary, men, SimLex):

    test=defaultdict(list)

    men_file=open(men).readlines()
    test['MEN']=[]
    for l in men_file:
        l2=l.replace('_N','').replace('_V','').replace('_A','').replace('\n','')
        j=l2.split(' ')
        test['MEN'].append(j)

    SimLex_file=open(SimLex).readlines()
    test['SimLex-999']=[]
    realfile=SimLex_file[1:]
    for i in realfile:
        j=i.split('\t')
        del j[2]
        test['SimLex-999'].append(j)

    if training_mode == 'W2V':
        vocabulary_words=vocabulary

    elif training_mode == 'RI' or training_mode == 'count':
        vocabulary_words = [word for word in vocabulary.keys()]
   
    else: 
        raise NameError('Possible training modes are: W2V, RI, count')

    for test_version in test.keys():

        gold=[]
        predicted=[]
        for j in test[test_version]:

            w1=j[0]
            w2=j[1]
            g=j[2]

            if w1 in vocabulary_words and w2 in vocabulary_words:

                if training_mode == 'W2V':
                    p=model.wv.similarity(w1,w2)

                elif training_mode == 'RI' or training_mode == 'count':
                    vector_w1 = numpy.array(model[vocabulary[w1]])[0]
                    vector_w2 = numpy.array(model[vocabulary[w2]])[0]
                    p = cosine_similarity(vector_w1, vector_w2)

                if p > 0:
                    gold.append(g)
                    predicted.append(p)
        print('\nResults for the model: {}'.format(training_mode))
        print("\n\tSpearman correlation for {}:".format(test_version))
        print('\t',score(gold, predicted, "spearman"))
        print("\n\tPearson correlation for {}:".format(test_version))
        print('\t',score(gold, predicted, "pearson"))

    #spearman=score(gold, predicted, "spearman")
    #pearson=score(gold, predicted, "pearson")
    #file.write('\n%s'%str(file) + "\n\tSpearman:\t%f"%spearman + "\n\tPearson:\t%f"%pearson)
