### EXAMPLE: python3 -m scripts.count_model --input_type wiki_${window} --filedir /mnt/cimec-storage-sata/users/andrea.bruera/wikiextractor/wiki_for_bert/wiki_clean_for_count.txt --window_size ${window}

import logging
import numpy
import collections
import spacy
import argparse
import dill as pickle
import os
import nonce2vec
import re
import tqdm
import scipy

from re import sub
from nonce2vec.utils.stopwords import stopwords
from nonce2vec.utils.count_based_models_utils import cosine_similarity, top_similarities, spacy_sentence_up, ReducedVocabulary, Corpus, clean_wikipedia, train_current_word
from collections import defaultdict
from numpy import zeros, vstack
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csr_matrix, save_npz

def initialize_count_model(args):

    cwd = os.getcwd()
    current_output_folder='{}/count_models/count_{}'.format(cwd, args.input_type)
    if args.write_to_file == True:
        try:
            os.makedirs(current_output_folder)
        except FileExistsError:
            logging.info("Training mode \"{}\" already found. Please change your training type".format(args.input_type))

    word_cooccurrences=defaultdict(lambda: defaultdict(int))

    return word_cooccurrences, current_output_folder

#def train_current_word(reduced_vocabulary, vocabulary, word_cooccurrences, args, corpus, word_index, current_word):

    ### Initialization of a new word: if the word has not been encountered yet, it is added to the vocabulary

    #current_word_index = vocabulary[current_word]

    #window_start = 1

    #if window_start <= args.window_size:

        ### Positive window word, notice the +

        #other_word_positive = corpus[word_index + window_start] 
        
        #if other_word_positive in reduced_vocabulary:
               
            #other_word_positive_index = vocabulary[other_word_positive]

            #word_cooccurrences[current_word_index][other_word_positive_index] += 1

        ### Symmetric negative window word, notice the -

        #other_word_negative = corpus[word_index - window_start] 

        #if other_word_negative in reduced_vocabulary:

            #other_word_negative_index = vocabulary[other_word_negative]

            #word_cooccurrences[current_word_index][other_word_negative_index] += 1

            ### Adding + 1, thus moving 1 further away from the word at hand

        #window_start += 1

    #return word_cooccurrences

        
logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)

parser=argparse.ArgumentParser()
parser.add_argument('--input_type', type=str, help = 'name of the dataset used for creating the word representations')
parser.add_argument('--filedir', type=str, help='Directory where the corpus files to be used for training are stored')
parser.add_argument('--window_size', type=int, default=5)
#parser.add_argument('--vector_size', type=int, default=1024, help='amount of words to be kept as columns of the matrix, according to their frequency')
parser.add_argument('--character', type=str, dest='character', required=False, default='house')
parser.add_argument('--write_to_file', required=False, action='store_true')
parser.add_argument('--print_results', required=False, action='store_true')
parser.add_argument('--number_similarities', type=int, default=20)
parser.add_argument('--spacy_sentence_up', type = bool, default = False)
parser.add_argument('--minimum_count', type=int, default=500)
parser.add_argument('--clean_wikipedia', required=False, type=str)
parser.add_argument('--load_files', action='store_true')

### Testing

args = parser.parse_args()

logging.info('Starting the basic count model training')

if args.clean_wikipedia:
    logging.info('Cleaning up Wikipedia')
    args.filedir = clean_wikipedia(args.filedir, args.clean_wikipedia)

training_parts = Corpus(args.filedir)

word_cooccurrences, current_output_folder = initialize_count_model(args)

if args.load_files:

    logging.info('Loading the vocabulary from file')
    vocabulary_file =open('{}/count_{}_vocabulary_trimmed.pickle'.format(current_output_folder, args.input_type), 'rb')
    vocabulary = pickle.load(vocabulary_file)
    vocabulary_file.close()

    #logging.info('Loading the cooccurrences from file')
    #word_cooccurrences_file = open('{}/count_{}_cooccurrences.pickle'.format(current_output_folder, args.input_type), 'rb')
    #word_cooccurrences = pickle.load(word_cooccurrences_file)
    #word_cooccurrences_file.close()
else:

    logging.info('Now creating the reduced vocabulary from scratch...')

    vocabulary = ReducedVocabulary(training_parts, args.minimum_count).to_dict()

    reduced_vocabulary = vocabulary.keys()

    logging.info('Reduced vocabulary size: {} words'.format(len(reduced_vocabulary)))

    logging.info('Count model initialized, started training...')

    if args.write_to_file:
        logging.info('Writing to file the reduced vocabulary, for quicker training in other trials')
        with open('{}/count_{}_vocabulary_trimmed.pickle'.format(current_output_folder, args.input_type), 'wb') as reduced_vocabulary_dict:
            pickle.dump(vocabulary, reduced_vocabulary_dict)

    for line in training_parts:

        #line_length = len(line)

        for word_index, word in enumerate(line):

            ### Collection of the window words which will be summed to the other ones
            if word in reduced_vocabulary:

                word_cooccurrences = train_current_word(reduced_vocabulary, vocabulary, word_cooccurrences, args, line, word_index, word)

if args.write_to_file:
    logging.info('Writing to file the cooccurrences vocabulary, for quicker training in other trials')
    with open('{}/count_{}_cooccurrences.pickle'.format(current_output_folder, args.input_type), 'wb') as cooccurrences_dict:
        pickle.dump(word_cooccurrences, cooccurrences_dict)

### Turning frequency counts into vectors

logging.info('Now creating the word vectors')

rows_sparse_matrix = []
columns_sparse_matrix = []
cells_sparse_matrix = []

for row_word_index in tqdm(word_cooccurrences): 
    for column_word_index in word_cooccurrences[row_word_index]:
        rows_sparse_matrix.append(row_word_index)
        columns_sparse_matrix.append(column_word_index)
        current_cooccurrence = word_cooccurrences[row_word_index][column_word_index]
        cells_sparse_matrix.append(current_cooccurrence)

shape_sparse_matrix = len(vocabulary)

logging.info('Now building the sparse matrix')

sparse_matrix_cooccurrences = csr_matrix((cells_sparse_matrix, (rows_sparse_matrix, columns_sparse_matrix)), shape = (shape_sparse_matrix, shape_sparse_matrix))

logging.info('Now saving to file the sparse matrix')

out_sparse_matrix = '{}/count_{}_sparse_matrix.npz'.format(current_output_folder, args.input_type)

scipy.sparse.save_npz(out_sparse_matrix, sparse_matrix_cooccurrences)

#word_vectors = {word_index : numpy.array(vector_as_list) for word_index, vector_as_list in word_vectors_as_lists.items()}

### Printing out top similarities for a query

#if args.print_results == True:
    #top_similarities(args.character, vocabulary, word_vectors, args.number_similarities)

### Saving to file

#logging.info('Now writing to file the word vectors')

#if args.write_to_file == True:

    #numpy.save('{}/count_{}_vectors.npy'.format(current_output_folder, args.input_type), word_vectors)

logging.info('Finished with training the background space for window size: {}'.format(args.window_size))
