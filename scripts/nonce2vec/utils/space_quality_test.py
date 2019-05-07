#Functions for calculating the MEN score for word embeddings created with Gensim W2V
import argparse
import pickle
import os
import gensim
import numpy as np
import collections

from collections import defaultdict

from scipy import stats

from gensim.models import Word2Vec

def cosine_similarity(vector_1, vector_2): 
    cosine_similarity = dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
    return cosine_similarity

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

def sim_check(args, model, men, SimLex):

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

    if args.training == 'W2V':
        vocabulary=model.wv.vocab.items()
        voc=[]
        for k,v in vocabulary:
            voc.append(k)

    if args.training == 'RI':
        voc = [word for word in RI_vocabulary.keys()]

    for test_version in test.keys():

        gold=[]
        predicted=[]
        for j in test[test_version]:

            w1=j[0]
            w2=j[1]
            g=j[2]

            if w1 in voc and w2 in voc:

                if args.training == 'W2V':
                    p=model.wv.similarity(w1,w2)

                elif args.training == 'RI':
                    index_w1 = RI_vocabulary[w1]
                    index_w2 = RI_vocabulary[w2]
                    p = cosine_similarity(model[index_w1], model[index_w2])

                if p > 0:
                    gold.append(g)
                    predicted.append(p)
        print('\nResults for the model: {}'.format(args.training))
        print("\n\tSpearman correlation for {}:".format(test_version))
        print('\t',score(gold, predicted, "spearman"))
        print("\n\tPearson correlation for {}:".format(test_version))
        print('\t',score(gold, predicted, "pearson"))

    #spearman=score(gold, predicted, "spearman")
    #pearson=score(gold, predicted, "pearson")
    #file.write('\n%s'%str(file) + "\n\tSpearman:\t%f"%spearman + "\n\tPearson:\t%f"%pearson)
