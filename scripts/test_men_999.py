###   EXAMPLE: python3 -m scripts.test_men_999 --training_mode count --directory count_models/count_wiki/ --men_999 men_999/
###   EXAMPLE 2: python3 -m scripts.test_men_999 W2V a men_999/ 

import os
import nonce2vec
import argparse
import gensim
import numpy
import dill as pickle
import scipy
import pdb
import logging

from scipy.sparse import csr_matrix, load_npz
from nonce2vec.utils.space_quality_test import *
from gensim.models import Word2Vec

parser=argparse.ArgumentParser()
parser.add_argument('--training_mode', type=str, help='Kind of model from which the word vectors were obtained. Possibilities: count, RI, W2V')
parser.add_argument('--directory', type=str, help='Directory where the files containing the word vectors are stored')
parser.add_argument('--men_999', type=str, help = 'Path to where the MEN and the SimLex999 dataset are stored')
parser.add_argument('--ppmi', action='store_true')

args=parser.parse_args()

if args.training_mode == 'W2V':
    model = Word2Vec.load('../wiki_training/data/wiki_w2v_2018_size400_window15_negative20_max_final_vocab250000_sg1')
    vocabulary=model.wv.vocab.items()
    voc=[]
    for k,v in vocabulary:
        voc.append(k)
if args.training_mode == 'count' or args.training_mode == 'RI':
    stored_files = os.walk(args.directory)
    for root, directory, training_files in stored_files:
        for single_file in training_files:
            if 'vocabulary' in single_file:
                vocabulary_file = open(os.path.join(root, single_file), 'rb')
                vocabulary = pickle.load(vocabulary_file)
            elif 'sparse' in single_file:
                model = load_npz(os.path.join(root, single_file))
                if args.ppmi:
                    model = ppmi(model)
                    #model = model.todense()
                else:
                    #model = load_npz(os.path.join(root, single_file)).todense()
                    #model = load_npz(os.path.join(root, single_file))
                    pass

men = '{}/men.txt'.format(args.men_999)
SimLex = '{}/SimLex-999.txt'.format(args.men_999)

sim_check(args.training_mode, model, vocabulary, men, SimLex)
