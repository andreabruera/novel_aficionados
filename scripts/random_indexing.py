### EXAMPLE: python3 -m scripts.random_indexing --write_to_file True --print_results True --non_zeros=16 wiki /mnt/cimec-storage-sata/users/andrea.bruera/wikiextractor/wiki_parts_small 10 2048


import numpy
import collections
import spacy
import argparse
import pickle
import os
import nonce2vec
import re

from collections import defaultdict
from re import sub
from nonce2vec.utils.stopwords import stopwords
from nonce2vec.utils.count_based_models_utils import cosine_similarity, top_similarities, spacy_sentence_up
from collections import defaultdict
from numpy import zeros, vstack


class Corpus(object):
    def __init__(self, args):
        self.filedir = args.filedir
        self.files = [os.path.join(root, name) for root, dirs, files in os.walk(args.filedir) for name in files]
        self.length = len(self.files)
        
    def __iter__(self):

        for individual_file in self.files: 

            training_lines = open(individual_file).readlines()
            if 'wiki' in args.input_type:
                training_part = [re.sub('\s+|\W+|_+|[0-9]+', ' ', line) for line in training_lines]  
            else:
                training_part = [re.sub('\s+|\W+|[0-9]+', ' ', line) for line in training_lines]  
            training_part = [line.strip(' ') for line in training_part if line != ' ']
            training_part = [word.lower() for line in training_part for word in line.split()]
            yield training_part

def create_word_index_vector(vocabulary, token, index_vectors, context_vectors):

    vocabulary[token]=len(vocabulary.keys())
    current_word_index=vocabulary[token]
    current_index_vector=numpy.zeros(args.vector_size)
    ones_indexes=numpy.random.choice(numpy.arange(args.vector_size), args.non_zeros, replace=False)
    middle_index = int(args.non_zeros / 2)
    negative_ones_indexes=ones_indexes[: middle_index -1]
    positive_ones_indexes=ones_indexes[middle_index :]
    for neg_index in negative_ones_indexes:
        current_index_vector[neg_index]=-1
    for pos_index in positive_ones_indexes:
        current_index_vector[pos_index]=1
    ### This condition checks whether this is the first token or not. If it is, it initializes the index_vectors array
    if len(index_vectors)>0:
        index_vectors=numpy.vstack((index_vectors, current_index_vector))
        context_vectors=numpy.vstack((index_vectors, current_index_vector))
    else:
        index_vectors=current_index_vector
        context_vectors=current_index_vector
    return vocabulary, index_vectors, context_vectors


def initialize_RI_model():

    vocabulary=defaultdict(str)
    index_vectors=[]
    context_vectors=[]
    return vocabulary, index_vectors, context_vectors

def train_current_word(vocabulary, index_vectors, context_vectors, args, corpus, word_index, word, stopwords):

    current_word_index=vocabulary[word]

    ### Collection of the window words which will be summed to the other ones
    if word_index >= args.window_size and (word_index + args.window_size + 1) <= len(corpus):

        window_words_negative = [token.strip('\n') for token in corpus[word_index - args.window_size : word_index - 1] if token.strip('\n') not in stopwords and token.strip('\n') != word]
        window_words_positive = [token.strip('\n') for token in corpus[word_index + 1 : word_index + args.window_size] if token.strip('\n') not in stopwords and token.strip('\n') != word]

        full_window = [word] + window_words_negative + window_words_positive

        if args.spacy_sentence_up == True:
            window_spacy = ' '.join(window)
            full_window = spacy_sentence_up(window_spacy)

        training_window = full_window[1:]
        word = full_window[0]

        for other_word in training_window:
 
            if other_word not in vocabulary.keys():
                vocabulary, index_vectors, context_vectors=create_word_index_vector(vocabulary, other_word, index_vectors, context_vectors)
            ### This condition checks whether this is the first token or not. If it is, it initializes the index_vectors array
            other_word_index=vocabulary[other_word]
            context_vectors[current_word_index] = numpy.add(context_vectors[current_word_index], index_vectors[other_word_index])

    return vocabulary, index_vectors, context_vectors

### Testing

parser=argparse.ArgumentParser()
parser.add_argument('input_type', type=str, help = 'name of the dataset used for creating the word representations')
parser.add_argument('filedir', type=str, help='Directory where the corpus files to be used for training are stored')
parser.add_argument('window_size', type=int, default=5)
parser.add_argument('vector_size', type=int, default=1024, help='amount of words to be kept as columns of the matrix, according to their frequency')
parser.add_argument('--character', type=str, default='house')
parser.add_argument('--write_to_file', type=bool, default=False)
parser.add_argument('--print_results', type=bool, default=True)
parser.add_argument('--number_similarities', type=int, default=20)
parser.add_argument('--non_zeros', type=int, default=16)
parser.add_argument('--spacy_sentence_up', type = bool, default = False)

args=parser.parse_args()

training_parts = Corpus(args)

training_length = training_parts.length

outputs = [round(value) for value in numpy.linspace(0, training_length, 100)]
outputs_dictionary = defaultdict(int)
for percentage, sentence_number in enumerate(outputs):
    outputs_dictionary[sentence_number] = percentage

if args.write_to_file == True:
    cwd=os.getcwd()
    try:
        current_output_folder='{}/RI_models/RI_{}'.format(cwd, args.input_type)
        os.makedirs(current_output_folder)
    except FileExistsError:
        print("Training mode \"{}\" already found. Please change your training type".format(args.input_type))

stopwords = stopwords()

vocabulary, index_vectors, context_vectors=initialize_RI_model()


for corpus_index, corpus in enumerate(training_parts):

    if corpus_index in outputs:
        print('Currently at {}% of the training\n'.format(outputs_dictionary[corpus_index] + 1)) 

    for word_index, word in enumerate(corpus):
        
        if word not in stopwords:
       
            ### Inizialization of a new word: if the word has not been encountered yet, its index vector is created and appended to the index_vectors matrix 

            if word not in vocabulary:
                vocabulary, index_vectors, context_vectors = create_word_index_vector(vocabulary, word, index_vectors, context_vectors)

            vocabulary, index_vectors, context_vectors = train_current_word(vocabulary, index_vectors, context_vectors, args, corpus, word_index, word, stopwords)


### Printing out top similarities for a query

if args.print_results == True:
    top_similarities(args.character, vocabulary, context_vectors, args.number_similarities)

### Saving to file

numpy.savez('{}/RI_{}_vectors'.format(current_output_folder, args.input_type), context_vectors, index_vectors)
with open('{}/RI_{}_vocabulary.pickle'.format(current_output_folder, args.input_type), 'wb') as RI_vocabulary:
    pickle.dump(vocabulary, RI_vocabulary)
