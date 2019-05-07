import numpy
import collections
import spacy

from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vector_1, vector_2): 
    fraction = norm(vector_1) * norm(vector_2)
    if fraction != 0:
        cosine_similarity = dot(vector_1, vector_2) / fraction
    else: 
        cosine_similarity = 0
    return cosine_similarity

def top_similarities(query, vocabulary, word_vectors, number_similarities):
 
    test_index=vocabulary[query]
    similarities_dictionary = defaultdict(float)

    for i, other_item in enumerate(vocabulary.keys()):
        character_sim=cosine_similarity(word_vectors[test_index], word_vectors[i])
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
