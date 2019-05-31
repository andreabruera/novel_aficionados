import nonce2vec 
import numpy 
import math
import sys
import os
import pickle
import torch
import collections

from nonce2vec.utils.novels_utilities import *
from nonce2vec.utils.count_based_models_utils import cosine_similarity, normalise
from numpy import dot,sqrt,sum,linalg
from math import sqrt
from torch import Tensor
from collections import defaultdict


class NovelsEvaluation:

    def __init__(self):
        self.reciprocal_ranks=[]
        self.ranks=[]
        self.disappearing_characters=[]

    def bert_evaluation(self, folder, characters_vectors, *layers_number, wiki_novel=False):
        if len(layers_number) > 1:
            chosen_characters_vectors = defaultdict(torch.Tensor)
            for character, layers in characters_vectors.items():
                for layer in layers_number:
                    if len(chosen_characters_vectors[character]) == 0:
                        chosen_characters_vectors[character] = layers[layer-1]
                    else:
                        chosen_characters_vectors[character] += layers[layer-1]
            string_layer_numbers = [str(number) for number in layers_number]
            layer_number = '_'.join(string_layer_numbers)
        else:
            layers_number = layers_number[0]
            layers_indices = range(layers_number)
            for layer in layers_indices: 
                chosen_characters_vectors = {character_and_part : layers[layer] for character_and_part, layers in characters_vectors.items()}
            layer_number = layers_number
        self.generic_evaluation(folder, chosen_characters_vectors, wiki_novel, layer_number)

    def generic_evaluation(self, folder, characters_vectors, wiki_novel=False, layer_number=None): 
        
        reciprocal_ranks = self.reciprocal_ranks
        ranks = self.ranks
        disappearing_characters = self.disappearing_characters

        if layer_number == None:
            if wiki_novel==False:
                evaluation_file=open('{}/evaluation_results.txt'.format(folder),'w')
                similarities_file=open('{}/similarities_results.txt'.format(folder), 'w')
            else:
                evaluation_file=open('{}/wiki_evaluation_results.txt'.format(folder),'w')
                similarities_file=open('{}/wiki_similarities_results.txt'.format(folder), 'w')
                
        else:
            if wiki_novel==False:
                folder = '{}/layer_{}'.format(folder, layer_number)
                os.makedirs(folder, exist_ok=True)
                evaluation_file=open('{}/evaluation_results.txt'.format(folder),'w')
                similarities_file=open('{}/similarities_results.txt'.format(folder), 'w')
            else:
                folder = '{}/layer_{}'.format(folder, layer_number)
                os.makedirs(folder, exist_ok=True)
                evaluation_file=open('{}/wiki_evaluation_results.txt'.format(folder),'w')
                similarities_file=open('{}/wiki_similarities_results.txt'.format(folder), 'w')
            

        for good_key, good_vector in characters_vectors.items():
            norm_good_vector=normalise(good_vector)
            character_name=good_key.split('_')[0]
            character_part=good_key.split('_')[1]
            marker = False

            for other_key, other_vector in characters_vectors.items():
                other_name=other_key.split('_')[0]
                other_part=other_key.split('_')[1]
                if other_part != character_part:
                    if other_name == character_name:
                        marker = True
            if marker == False:
                disappearing_characters.append(character_name)

        for good_key, good_vector in characters_vectors.items():
            norm_good_vector=normalise(good_vector)
            character_name=good_key.split('_')[0]
            character_part=good_key.split('_')[1]
            if character_name not in disappearing_characters:
                simil_dict={}

                for other_key, other_vector in characters_vectors.items():
                    other_name=other_key.split('_')[0]
                    other_part=other_key.split('_')[1]
                    if other_part != character_part and other_name not in disappearing_characters:
                        norm_other_vector=normalise(other_vector)
                        simil=cosine_similarity(norm_good_vector, norm_other_vector)
                        simil_dict[float(simil)]=other_key
                sorted_simil_list=sorted(simil_dict,reverse=True)
                for rank, similarity in enumerate(sorted_simil_list):
                    rank+=1
                    current_character=simil_dict[similarity].split('_')
                    if current_character[0] == character_name:
                        reciprocal_ranks.append(1/rank)
                        ranks.append(rank)
                        similarities_file.write('\nResult for the vector: {}, coming from part {} of the book\nRank: {} out of {} characters\nReciprocal rank: {}\nCosine similarity to the query: {}\n\n'.format(character_name, character_part, rank, len(sorted_simil_list), (1/rank), similarity))
                        for i in sorted_simil_list:
                            similarities_file.write('{} - {}\n'.format(simil_dict[i], i))

        MRR=numpy.mean(reciprocal_ranks)
        median_rank=numpy.median(ranks)
        mean_rank=numpy.mean(ranks)

        evaluation_file.write('MRR:\t{}\nMedian rank:\t{}\nMean rank:\t{}\nTotal number of characters considered:\t{}\n'.format(MRR, median_rank, mean_rank, len(sorted_simil_list)+1)) 

