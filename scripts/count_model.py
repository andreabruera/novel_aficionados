### EXAMPLE: python3 -m scripts.count_model --write_to_file True --print_results False wiki /mnt/cimec-storage-sata/users/andrea.bruera/wikiextractor/wiki_parts 10 5000


import numpy
import collections
import spacy
import argparse
import pickle
import os
import nonce2vec
import re

from re import sub
from nonce2vec.utils.stopwords import stopwords
from nonce2vec.utils.count_based_models_utils import cosine_similarity, top_similarities, spacy_sentence_up
from collections import defaultdict
from numpy import zeros, vstack


def initialize_count_model():

    vocabulary = defaultdict(int)
    word_counters = defaultdict(int)
    word_cooccurrences=defaultdict(int)
    return vocabulary, word_counters, word_cooccurrences

def train_current_word(vocabulary, word_counters, word_cooccurrences, args, corpus, word_index, word, stopwords):

    ### Collection of the window words which will be summed to the other ones
    if word_index >= args.window_size and (word_index + args.window_size +1) <= len(corpus):

        negative_indices = numpy.arange(word_index - args.window_size, word_index)
        positive_indices = numpy.arange(word_index + 1, word_index + args.window_size + 1)

        window_words_negative = [corpus[index].strip('\n') for index in negative_indices if corpus[index].strip('\n') not in stopwords and  corpus[index].strip('\n') != word]
        window_words_positive = [corpus[index].strip('\n') for index in positive_indices if corpus[index].strip('\n') not in stopwords and  corpus[index].strip('\n') != word]

        current_sentence= [word] + window_words_negative + window_words_positive
        current_word = current_sentence[0]
        other_words = current_sentence[1:]

        for other_word in other_words:

            word_cooccurrences['{} {}'.format(current_word, other_word)] += 1

    return vocabulary, word_counters, word_cooccurrences

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
                
        
        


### Testing

parser=argparse.ArgumentParser()
parser.add_argument('input_type', type=str, help = 'name of the dataset used for creating the word representations')
parser.add_argument('filedir', type=str, help='Absolute path to directory containing the files to be used for training')
parser.add_argument('window_size', type=int, default=5)
parser.add_argument('vector_size', type=int, default=1024, help='amount of words to be kept as columns of the matrix, according to their frequency')
parser.add_argument('--character', type=str, default='house')
parser.add_argument('--write_to_file', type=bool, default=False)
parser.add_argument('--print_results', type=bool, default=True)
parser.add_argument('--number_similarities', type=int, default=20)
parser.add_argument('--spacy_sentence_up', type = bool, default = False)

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
        print("Training mode \"{}\" already found. Please change your training type".format(args.input_type))

stopwords = stopwords()

vocabulary, word_counters, word_cooccurrences=initialize_count_model()
print('Count model initialized, started training...')

for corpus_index, corpus in enumerate(training_parts):

    if corpus_index in outputs:
        print('Currently at {}% of the training\n'.format(outputs_dictionary[corpus_index] + 1)) 

    for word_index, word in enumerate(corpus):

        if word not in stopwords:

            ### Inizialization of a new word: if the word has not been encountered yet, it is added to the vocabulary
            if word not in vocabulary.keys():
                vocabulary[word]=len(vocabulary.keys())

            word_counters[word] += 1


            vocabulary, word_counters, word_cooccurrences = train_current_word(vocabulary, word_counters, word_cooccurrences, args, corpus, word_index, word, stopwords)


### Selecting the most common words

columns = [pairing[0] for pairing in sorted(word_counters.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) if pairing[1] not in stopwords][: args.vector_size]

### Turning frequency counts into vectors

for row_index, row_word in enumerate(vocabulary.keys()):

    if row_index == 0:
        word_vectors = numpy.zeros(args.vector_size)
    else:
        current_vector = numpy.zeros(args.vector_size)
        word_vectors = numpy.vstack((word_vectors, current_vector))

    if row_index in outputs:
        print('Currently at {}% in the creation of the vectors...\n'.format(outputs_dictionary[row_index]))

    for column_index, column_word in enumerate(columns):
        current_cooccurrence = '{} {}'.format(row_word, column_word) 
        if current_cooccurrence in word_cooccurrences.keys():
            
            if row_index == 0:
                word_vectors[column_index] = word_cooccurrences[current_cooccurrence]
            else:
                word_vectors[row_index][column_index] = word_cooccurrences[current_cooccurrence]

### Printing out top similarities for a query

if args.print_results == True:
    top_similarities(args.character, vocabulary, word_vectors, args.number_similarities)

### Saving to file

if args.write_to_file == True:

    numpy.save('{}/count_{}_vectors.npy'.format(current_output_folder, args.input_type), word_vectors)

    with open('{}/count_{}_vocabulary.pickle'.format(current_output_folder, args.input_type), 'wb') as count_vocabulary:
        pickle.dump(vocabulary, count_vocabulary)

    with open('{}/count_{}_word_cooccurrences.pickle'.format(current_output_folder, args.input_type), 'wb') as cooccurrences_dictionary:
        pickle.dump(word_cooccurrences, cooccurrences_dictionary)

    with open('{}/count_{}_word_frequencies.pickle'.format(current_output_folder, args.input_type), 'wb') as frequencies_dictionary:
        pickle.dump(word_counters, frequencies_dictionary)
