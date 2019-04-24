import numpy
import collections
import spacy
import argparse

from collections import defaultdict
from numpy import zeros, vstack
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vector_1, vector_2): 
    cosine_similarity = dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
    return cosine_similarity

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

def clean_sentence_up(sentence):

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

def initialize_RI_model():

    vocabulary=defaultdict(str)
    index_vectors=[]
    context_vectors=[]
    return vocabulary, index_vectors, context_vectors

def train_current_word(vocabulary, index_vectors, context_vectors, args, clean_sentence, index, token):

    current_word_index=vocabulary[token]
    ### Collection of the window words which will be summed to the other ones
    if index > args.window_size:
        window_words_negative = [token for token in clean_sentence[index - args.window_size : index-1]]
        window_words_positive = [token for token in clean_sentence[index+1 : index + args.window_size]]
        for other_word in window_words_negative: 
            if other_word not in vocabulary.keys():
                vocabulary, index_vectors, context_vectors=create_word_index_vector(vocabulary, other_word, index_vectors, context_vectors)
            ### This condition checks whether this is the first token or not. If it is, it initializes the index_vectors array
            other_word_index=vocabulary[other_word]
            context_vectors[current_word_index]=numpy.add(context_vectors[current_word_index], context_vectors[other_word_index])
        for other_word in window_words_positive: 
            if other_word not in vocabulary.keys():
                vocabulary, index_vectors, context_vectors=create_word_index_vector(vocabulary, other_word, index_vectors, context_vectors)
            other_word_index=vocabulary[other_word]
            context_vectors[current_word_index]=numpy.add(context_vectors[current_word_index], context_vectors[other_word_index])

    return vocabulary, index_vectors, context_vectors



### Testing

parser=argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the file to be used for training')
parser.add_argument('window_size', type=int, default=5)
parser.add_argument('vector_size', type=int, default=1024)
parser.add_argument('non_zeros', type=int, default=16)
parser.add_argument('character', type=str, default='house')

args=parser.parse_args()

spacy_language_model = spacy.load("en_core_web_sm")

full_text=[line.strip('\n') for line in open(args.file).readlines()]

vocabulary, index_vectors, context_vectors=initialize_RI_model()

for sentence in full_text:

    clean_sentence=clean_sentence_up(sentence)

    for index, token in enumerate(clean_sentence):
       
        ### Inizialization of a new word: if the word has not been encountered yet, its index vector is created and appended to the index_vectors matrix 
        if token not in vocabulary:
            vocabulary, index_vectors, context_vectors=create_word_index_vector(vocabulary, token, index_vectors, context_vectors)

        vocabulary, index_vectors, context_vectors = train_current_word(vocabulary, index_vectors, context_vectors, args, clean_sentence, index, token)


### Printing out some virtually random results

test_index=vocabulary[args.character]
for i, other_item in enumerate(vocabulary.keys()):
    clean_sentence_sim=cosine_similarity(context_vectors[test_index], context_vectors[i])
    if clean_sentence_sim > 0.2:
        print('Somiglianza massima tra {} e {}: {}'.format(args.character, other_item, clean_sentence_sim))

