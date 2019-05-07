### EXAMPLE: python3 -m scripts.count_model --write_to_file True --logging.info_results False wiki /mnt/cimec-storage-sata/users/andrea.bruera/wikiextractor/wiki_parts 10 5000

import logging
import numpy
import collections
import spacy
import argparse
import pickle
import os
import nonce2vec
import re
import tqdm

import dill as pickle
from re import sub
from nonce2vec.utils.stopwords import stopwords
from nonce2vec.utils.count_based_models_utils import cosine_similarity, top_similarities, spacy_sentence_up
from collections import defaultdict
from numpy import zeros, vstack


def initialize_count_model():

    vocabulary = defaultdict(int)
    word_counters = defaultdict(int)
    word_cooccurrences=defaultdict(lambda: defaultdict(int))
    soon_to_become_past_sentence = []
    return vocabulary, word_counters, word_cooccurrences, soon_to_become_past_sentence

def train_current_word(vocabulary, word_counters, word_cooccurrences, args, corpus, word_index, word, stopwords):

    current_word = word

    ### Inizialization of a new word: if the word has not been encountered yet, it is added to the vocabulary
    if current_word not in vocabulary.keys():
        vocabulary[current_word]=len(vocabulary.keys())

    word_counters[current_word] += 1

    current_word_index = vocabulary[current_word]

    window_start = 1

    while window_start <= args.window_size:

        ### Positive window word, notice the +

        other_word_positive = corpus[word_index + window_start] 

        if other_word_positive not in vocabulary.keys():
            vocabulary[other_word_positive] = len(vocabulary.keys())
           
        other_word_positive_index = vocabulary[other_word_positive]

        word_cooccurrences[current_word_index][other_word_positive_index] += 1

        ### Symmetric negative window word, notice the -

        other_word_negative = corpus[word_index - window_start] 

        if other_word_negative not in vocabulary.keys():
            vocabulary[other_word_negative] = len(vocabulary.keys())
           
        other_word_negative_index = vocabulary[other_word_negative]

        word_cooccurrences[current_word_index][other_word_negative_index] += 1

        ### Adding + 1, thus moving 1 further away from the word at hand

        window_start += 1

    return vocabulary, word_counters, word_cooccurrences

class Corpus(object):
    def __init__(self, args):
        self.filedir = args.filedir
        self.files = [os.path.join(root, name) for root, dirs, files in os.walk(args.filedir) for name in files]
        self.length = len(self.files)
        
    def __iter__(self):

        for individual_file in self.files: 
            logging.info(individual_file)

            training_lines = open(individual_file).readlines()
            length = len(training_lines)
            #training_part = [line.strip(' ') for line in training_part if line != ' ']
            #training_part = [word.lower() for line in training_part for word in line.split()]
            #yield training_part
            for line in training_lines:
                line = re.sub('\W+|_+|[0-9]+', ' ', line)  
                line = re.sub('\s+', ' ', line)  
                line = [word.lower() for word in line.split()]
                yield line, length
                
        
logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)


### Testing

parser=argparse.ArgumentParser()
parser.add_argument('input_type', type=str, help = 'name of the dataset used for creating the word representations')
parser.add_argument('filedir', type=str, help='Absolute path to directory containing the files to be used for training')
parser.add_argument('window_size', type=int, default=5)
parser.add_argument('vector_size', type=int, default=1024, help='amount of words to be kept as columns of the matrix, according to their frequency')
parser.add_argument('--character', type=str, default='house')
parser.add_argument('--write_to_file', type=bool, default=False)
parser.add_argument('--logging.info_results', type=bool, default=True)
parser.add_argument('--number_similarities', type=int, default=20)
parser.add_argument('--spacy_sentence_up', type = bool, default = False)
parser.add_argument('--minimum_count', type = int, default = 300)

args=parser.parse_args()

training_parts = Corpus(args)

training_length = training_parts.length

outputs = [round(value) for value in numpy.linspace(0, training_length, 100)]
outputs_dictionary = defaultdict(int)
for percentage, sentence_number in enumerate(outputs):
    outputs_dictionary[sentence_number] = percentage

if args.write_to_file == True:
    cwd = os.getcwd()
    try:
        current_output_folder='{}/count_models/count_{}'.format(cwd, args.input_type)
        os.makedirs(current_output_folder)
    except FileExistsError:
        logging.info("Training mode \"{}\" already found. Please change your training type".format(args.input_type))

stopwords = stopwords()

vocabulary, word_counters, word_cooccurrences, soon_to_become_past_sentence = initialize_count_model()
logging.info('Count model initialized, started training...')

sentence_index = 0

for corpus, length in training_parts:

    sentence_index += 1
    step = length / 100
    length_dictionaries = { v : k for k, v in enumerate(range(1, round((length/100)*100), round(length / 100)))}
    if sentence_index in length_dictionaries.keys():
        logging.info('Current at {} / 100 of the training'.format(length_dictionaries[sentence_index])) 

    corpus_length = len(corpus)
    #logging.info(corpus)

    for word_index, word in enumerate(corpus):

        ### Collection of the window words which will be summed to the other ones
        if word_index >= args.window_size and (word_index + args.window_size +1) <= corpus_length:

            vocabulary, word_counters, word_cooccurrences = train_current_word(vocabulary, word_counters, word_cooccurrences, args, corpus, word_index, word, stopwords)

#if args.write_to_file == True:
    #with open('{}/count_{}_vocabulary.pickle'.format(current_output_folder, args.input_type), 'wb') as count_vocabulary:
        #pickle.dump(vocabulary, count_vocabulary)

    #with open('{}/count_{}_word_cooccurrences.pickle'.format(current_output_folder, args.input_type), 'wb') as cooccurrences_dictionary:
        #pickle.dump(word_cooccurrences, cooccurrences_dictionary)

    #with open('{}/count_{}_word_frequencies.pickle'.format(current_output_folder, args.input_type), 'wb') as frequencies_dictionary:
        #pickle.dump(word_counters, frequencies_dictionary)

### Selecting the most common words for the columns

trimmed_vocabulary = [freq_tuple[0] for freq_tuple in sorted(word_counters.items(), key = lambda kv: (kv[1], kv[0]), reverse = True) if freq_tuple[1] >= args.minimum_count and freq_tuple[0] not in stopwords()]

### Taking away words if they don't reach a minimum count in the corpus
#rows_vocabulary = {word : key for word, key in vocabulary.items() if word_counters[word] >= args.minimum_count}

### Turning frequency counts into vectors

word_vectors_as_lists = defaultdict(list)

for row_word, row_word_index in rows_vocabulary.items(): 
    word_vectors_as_lists[row_word] = [word_cooccurrences[row_word_index][column_word_index] for word, column_word_index in columns_vocabulary.items()]

word_vectors = {word : numpy.array(vector_as_list) for word, vector_as_list in word_vectors_as_lists.items()}

### Printing out top similarities for a query

if args.print_results == True:
    top_similarities(args.character, rows_vocabulary, word_vectors, args.number_similarities)

### Saving to file

if args.write_to_file == True:

    numpy.save('{}/count_{}_vectors.npy'.format(current_output_folder, args.input_type), word_vectors)

    with open('{}/count_{}_vocabulary_trimmed.pickle'.format(current_output_folder, args.input_type), 'wb') as count_vocabulary_trimmed:
        pickle.dump(vocabulary_trimmed, count_vocabulary_trimmed)

    with open('{}/count_{}_columns.pickle'.format(current_output_folder, args.input_type), 'wb') as columns_pickle:
        pickle.dump(columns_pickle, columns)
