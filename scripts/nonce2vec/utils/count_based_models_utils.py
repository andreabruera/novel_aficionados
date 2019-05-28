import numpy
import collections
import spacy
import math
import tqdm
import os
import re
import logging
import argparse
import pickle
import scipy

from scipy.sparse import csr_matrix, load_npz
from collections import defaultdict
from numpy import dot, sum
from numpy.linalg import norm
from math import sqrt
from tqdm import tqdm

def normalise(vector):
    norm_vector = norm(vector)
    if norm_vector == 0:
        return vector
    vector = vector / norm_vector
    #print(sum([i*i for i in v]))
    return vector

#def norm(value):
    #v=float(value)
    #norm=v / sqrt((sum(v**2)))
    #return float(norm)

def cosine_similarity(vector_1, vector_2): 
    if len(vector_1) != len(vector_2):
        raise ValueError('Vectors must be of same length')
    vector_1 = numpy.squeeze(vector_1)
    vector_2 = numpy.squeeze(vector_2)
    denominator_a = numpy.dot(vector_1, vector_1)
    denominator_b = numpy.dot(vector_2, vector_2)
    denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
    if float(denominator) == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot(vector_1, vector_2) / denominator
    return cosine_similarity

def top_similarities(query, vocabulary, word_vectors, number_similarities):
    
    test_index=vocabulary[query]
    similarities_dictionary = defaultdict(float)

    for other_item, other_index in vocabulary.items():
        character_sim=cosine_similarity(word_vectors[test_index], word_vectors[other_index])
        if character_sim > 0.01:
            similarities_dictionary[other_item] = character_sim

    similarities_dictionary = [sim for sim in sorted(similarities_dictionary.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)][: number_similarities]
    print(similarities_dictionary)

def spacy_setup():
    spacy_language_model = spacy.load("en_core_web_sm")
    return spacy_language_model

def spacy_sentence_up(sentence):

    sentence_spacy=spacy_language_model(sentence)

    sentence_clean=[token.lemma_ for token in sentence_spacy if ((
    token.pos_ == 'VERB' or 
    token.pos_ == 'NOUN' or 
    token.pos_ == 'PROPN' or 
    token.pos_ == 'ADJ'
    ) 
    and (
    token.lemma_ != '-PRON-' and 
    token.lemma_ != 'have' and 
    token.lemma_ != 'be' and
    token.lemma_ != 'could' and
    token.lemma_ != 'make' and
    token.lemma_ != 'do' and
    token.lemma_ != 'not'
    )
    and (
    'W' not in token.tag_ and
    'PR' not in token.tag_
    )
    )]

    return sentence_clean
    return sentence_clean

def clean_wikipedia(path_to_wikipedia, path_to_output):
    logging.info('Opening the output file: {}'.format(path_to_output))
    with open(path_to_output, 'w') as out:
        logging.info('Opening the input file: {}'.format(path_to_wikipedia))
        with open(path_to_wikipedia) as wiki_file:
            wiki_lines=wiki_file.readlines()
            for line in tqdm(wiki_lines):
                line = re.sub('\s+|\W+|_+|[0-9]+', ' ', line)
                line = re.sub('\s+', ' ', line)
                if line != ' ' and line != '' and len(line)>50:
                    out.write(line.lower())
                    out.write('\n')  
    return path_to_output

class ReducedVocabulary:
    def __init__(self, corpus, minimum_count):
        self.word_counters = defaultdict(int)
        self.reduced_vocabulary = defaultdict(int)
        self.minimum_count = minimum_count
        self.corpus = corpus
        self.trim()

    def count(self):
        logging.info('Counting the words in the corpus, so as to trim the vocabulary size')
        for line in tqdm(self.corpus):
            for word in line:
                self.word_counters[word] += 1

    def trim(self):
        self.count()
        logging.info('Trimming the vocabulary')
        word_counters = self.word_counters
        for word, frequency in tqdm(word_counters.items()):
            if frequency > self.minimum_count:
                self.reduced_vocabulary[word] = len(self.reduced_vocabulary.keys())
        return self.reduced_vocabulary

    def to_dict(self):
        return self.reduced_vocabulary

            

class Corpus(object):
    def __init__(self, filedir):
        self.filedir = filedir
        self.files = [os.path.join(root, name) for root, dirs, files in os.walk(filedir) for name in files if '.txt' in name]
        print(self.files)
        
    def __iter__(self):

        for individual_file in self.files: 
            training_lines = open(individual_file).readlines()
            for line in tqdm(training_lines):
                line = line.strip().split()
                yield line

def train_current_word(reduced_vocabulary, vocabulary, word_cooccurrences, args, sentence, word_index, current_word):

    ### Initialization of a new word: if the word has not been encountered yet, it is added to the vocabulary

    sentence_length = len(sentence)
    current_word_index = vocabulary[current_word]

    window = range(1, args.window_size+1)
    ### Positive window word, notice the +

    if word_index < args.window_size or word_index < (sentence_length - args.window_size):

        for position in window:
            next_position = word_index + position

            if next_position < len(sentence):

                other_word_positive = sentence[next_position] 

                if other_word_positive in reduced_vocabulary:
                       
                    other_word_positive_index = vocabulary[other_word_positive]

                    word_cooccurrences[current_word_index][other_word_positive_index] += 1

    ### Symmetric negative window word, notice the -
    if word_index >= args.window_size:
        for position in window:
            next_position = word_index - position

            if next_position >= 0:

                other_word_negative = sentence[next_position] 

                if other_word_negative in reduced_vocabulary:

                    other_word_negative_index = vocabulary[other_word_negative]

                    word_cooccurrences[current_word_index][other_word_negative_index] += 1

    return word_cooccurrences

