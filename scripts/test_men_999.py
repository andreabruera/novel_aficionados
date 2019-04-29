###   EXAMPLE: python3 -m scripts.test_men_999 RI RI_wiki men_999/
###   EXAMPLE 2: python3 -m scripts.test_men_999 W2V a men_999/ 

import os
import nonce2vec
import argparse
import gensim

from nonce2vec.utils.space_quality_test import *
from gensim.models import Word2Vec

parser=argparse.ArgumentParser()
parser.add_argument('training', type=str, help='Kind of model from which the word vectors were obtained. Possibilities: count, RI, W2V')
parser.add_argument('directory', type=str, help='Directory where the files containing the word vectors are stored')
parser.add_argument('men_999', type=str, help = 'Path to where the MEN and the SimLex999 dataset are stored')

args=parser.parse_args()

if args.training == 'W2V':
    model = Word2Vec.load('../wiki_training/data/wiki_w2v_2018_size400_window15_negative20_max_final_vocab250000_sg1')

elif args.training == 'RI':
    vectors_directory = os.listdir(args.directory)
    numpy_vectors = [numpy_arrays for numpy_arrays in vectors_directory if 'vectors.npy' in numpy_arrays][0]
    all_vectors = numpy.load(numpy_vectors)
    model=all_vectors['context_vectors']
    pickle_vocabulary_filename=[pickle_file for pickle_file in vectors_directory if '.pickle' in pickle_file][0]
    with open('{}/{}'.format(args.directory, pickle_vocabulary_filename, 'rb')) as pickle_vocabulary:
        RI_vocabulary = pickle.load(pickle_vocabulary)

men = '{}/men.txt'.format(args.men_999)
SimLex = '{}/SimLex-999.txt'.format(args.men_999)

sim_check(args, model, men, SimLex)
