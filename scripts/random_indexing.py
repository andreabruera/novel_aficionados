### EXAMPLE: python3 -m scripts.random_indexing --input_type wiki --filedir /mnt/cimec-storage-sata/users/andrea.bruera/wiki_training/data/clean --window_size 10 --vector_size 2048


import numpy
import collections
import spacy
import argparse
import dill as pickle
import os
import nonce2vec
import re
import scipy
import tqdm
import logging

from collections import defaultdict
from re import sub
from nonce2vec.utils.stopwords import stopwords
from nonce2vec.utils.count_based_models_utils import cosine_similarity, top_similarities, spacy_sentence_up, ReducedVocabulary, Corpus, clean_wikipedia
from collections import defaultdict
from numpy import zeros, vstack
from scipy.sparse import csr_matrix
from tqdm import tqdm


def create_word_index_vector(vocabulary, token, index_vectors, context_vectors):

    vocabulary[token]=len(vocabulary.keys())
    current_word_index=vocabulary[token]
    
    columns=[int(number) for number in numpy.random.choice(numpy.arange(args.vector_size), args.non_zeros, replace=False)]
    row = numpy.zeros(args.non_zeros)
    onez = []
    for column_index, column_position in enumerate(columns):
        if column_index < int(round(args.non_zeros/2)):
            onez.append(1)
        else:
            onez.append(-1)
    current_index_vector = csr_matrix((onez, (row, columns)), shape=(1, args.vector_size), dtype=int)
    ### This condition checks whether this is the first token or not. If it is, it initializes the index_vectors array

    #if len(index_vectors)>0:
        #index_vectors=numpy.vstack((index_vectors, current_index_vector))
        #context_vectors=numpy.vstack((index_vectors, current_index_vector))
    #else:
        #index_vectors=current_index_vector
        #context_vectors=current_index_vector
    index_vectors[current_word_index] = current_index_vector.toarray()
    context_vectors[current_word_index] = current_index_vector.toarray()

    return vocabulary, index_vectors, context_vectors


def initialize_RI_model(vocabulary):


    cwd = os.getcwd()
    current_output_folder='{}/RI_models/count_{}'.format(cwd, args.input_type)
    if args.write_to_file == True:
        try:
            os.makedirs(current_output_folder)
        except FileExistsError:
            logging.info("Training mode \"{}\" already found. Please change your training type".format(args.input_type))

    index_vectors=defaultdict(numpy.ndarray)
    context_vectors = defaultdict(numpy.ndarray)
    for word in reduced_vocabulary:
        if word not in vocabulary.keys():
            vocabulary, index_vectors, context_vectors = create_word_index_vector(vocabulary, word, index_vectors, context_vectors)
    return index_vectors, context_vectors, current_output_folder

def train_current_word(vocabulary, index_vectors, context_vectors, args, corpus, word_index, current_word):

    current_word_index=vocabulary[word]

    ### Collection of the window words which will be summed to the other ones
    if word_index >= args.window_size and (word_index + args.window_size + 1) <= len(corpus):

        current_word_index = vocabulary[current_word]

        window_start = 1

        if window_start <= args.window_size:

            ### Positive window word, notice the +

            other_word_positive = corpus[word_index + window_start] 
            
            if other_word_positive in reduced_vocabulary:
                   
                other_word_positive_index = vocabulary[other_word_positive]
                import pdb; pdb.set_trace()

                for col_index, col_value in enumerate(context_vectors[current_word_index]):
                    context_vectors[current_word_index][col_index] += index_vectors[other_word_index][col_index]    

            ### Symmetric negative window word, notice the -

            other_word_negative = corpus[word_index - window_start] 

            if other_word_negative in reduced_vocabulary:

                other_word_negative_index = vocabulary[other_word_negative]

                for col_index, col_value in enumerate(context_vectors[current_word_index]):
                    context_vectors[current_word_index][col_index] += index_vectors[other_word_index][col_index]    

                ### Adding + 1, thus moving 1 further away from the word at hand

            window_start += 1
                
    return context_vectors

### Testing

logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)

parser=argparse.ArgumentParser()
parser.add_argument('--input_type', type=str, help = 'name of the dataset used for creating the word representations')
parser.add_argument('--filedir', type=str, help='Directory where the corpus files to be used for training are stored')
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--vector_size', type=int, default=1024, help='amount of words to be kept as columns of the matrix, according to their frequency')
parser.add_argument('--character', type=str, dest='character', required=False, default='house')
parser.add_argument('--write_to_file', required=False, action='store_true')
parser.add_argument('--print_results', required=False, action='store_true')
parser.add_argument('--number_similarities', type=int, default=20)
parser.add_argument('--non_zeros', dest='non_zeros', type=int, default=16)
parser.add_argument('--spacy_sentence_up', type = bool, default = False)
parser.add_argument('--minimum_count', type=int, default=500)
parser.add_argument('--clean_wikipedia', required=False, type=str)
parser.add_argument('--load_files', action='store_true', default=False)

args=parser.parse_args()

logging.info('Starting the basic Random Indexing model training')

if args.clean_wikipedia:
    logging.info('Cleaning up Wikipedia')
    args.filedir = clean_wikipedia(args.filedir, args.clean_wikipedia)

training_parts = Corpus(args.filedir)

number_of_files = training_parts.number_of_files

if args.load_files:

    logging.info('Loading the vocabulary from file')
    vocabulary_file =open('{}/count_{}_vocabulary_trimmed.pickle'.format(current_output_folder, args.input_type), 'rb')
    vocabulary = pickle.load(vocabulary_file)
    vocabulary_file.close()
else:
    logging.info('Now creating the reduced vocabulary from scratch...')

    vocabulary = ReducedVocabulary(training_parts, args.minimum_count).to_dict()

reduced_vocabulary = vocabulary.keys()

index_vectors, context_vectors, current_output_folder = initialize_RI_model(vocabulary)

logging.info('Reduced vocabulary size: {} words'.format(len(reduced_vocabulary)))

if args.write_to_file:
    with open('{}/RI_{}_vocabulary.pickle'.format(current_output_folder, args.input_type), 'wb') as RI_vocabulary:
        pickle.dump(vocabulary, RI_vocabulary)

logging.info('Random Indexing model initialized, started training...')

for line in training_parts:

    line_length = len(line)

    for word_index, word in enumerate(line):

        if word in reduced_vocabulary:
       
            ### Inizialization of a new word: if the word has not been encountered yet, its index vector is created and appended to the index_vectors matrix 

            context_vectors = train_current_word(vocabulary, index_vectors, context_vectors, args, line, word_index, word)

### Printing out top similarities for a query

if args.print_results == True:
    top_similarities(args.character, vocabulary, context_vectors, args.number_similarities)

### Saving to file

if args.write_to_file:

    numpy.savez('{}/RI_{}_vectors'.format(current_output_folder, args.input_type), context_vectors, index_vectors)
