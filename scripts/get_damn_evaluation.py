import nonce2vec 
import numpy 
import math
import sys
import os
import pickle

from nonce2vec.utils.novels_utilities import *
from numpy import dot,sqrt,sum,linalg
from math import sqrt

def normalise(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    v = v / norm
    #print(sum([i*i for i in v]))
    return v

def norm(value):
    v=float(value)
    norm=v / numpy.sqrt((numpy.sum(v**2)))
    return float(norm)

def _cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = numpy.dot(peer_v, query_v)
    den_a = numpy.dot(peer_v, peer_v)
    den_b = numpy.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))

def extract_char_vectors(book, training):
    novels_list=utilities.get_books_list(book)
    char_list=utilities.get_characters_list(book)
    os.makedirs('{}_training'.format(training),exist_ok=True)
    os.makedirs('{}_training/data_{}_training'.format(training, training),exist_ok=True)
    books_vectors=['{}.character_vectors'.format(i) for i in novels_list]
    vectors_dict={}
    for version in books_vectors:
            for index, character2 in enumerate(char_list):
                character=character2.replace(' ','_')
                vectors_dict['{}_{}'.format(character, version)]=[]
                f=open('{}_training/data_{}_training/{}'.format(training,training,version)).readlines()
                marker=False
                for line in f:
                    f2=line.strip('\b').strip('\n').strip('\b').replace('uncle podger','uncle_podger').replace(' ','\t').split('\t')
                    f3=f2[1:]
                    if marker==True:
                        if index!=len(char_list)-1:
                            if f2[0]!=char_list[index+1].replace(' ','_'):
                                for w in f3:
                                    if w!=']' and len(w)>0:
                                        w2=w.replace('[','').replace(']','')
                                        vectors_dict['{}_{}'.format(character, version)].append(w2)
                            elif marker==True and f2[0]==char_list[index+1].replace(' ','_'):
                                marker=False
                                break
                        else:
                            for w in f3:
                                if w=='[' or w==']' or len(w)==0:
                                    pass
                                else:
                                    if len(w)>0:
                                        w2=w.replace(']','').replace('[','')
                                        vectors_dict['{}_{}'.format(character, version)].append(w2)
                    elif marker==False and f2[0]==character:
                        marker=True
                        for w in f3:
                            if w=='[':
                                pass
                            else:
                                if len(w)>0:
                                    w2=w.replace('[','').replace(']','')
                                    vectors_dict['{}_{}'.format(character, version)].append(w2)
    index=len(vectors_dict)/3
    return vectors_dict,index
'''
characters_vectors,index=extract_char_vectors(sys.argv[1],sys.argv[2])
'''
folder=sys.argv[1]
number=sys.argv[2]

char_list=get_characters_list(folder, number)
#folder=sys.argv[2]
characters_vectors=pickle.load(open('{}/data_output/{}.pickle'.format(folder, number), 'rb'))
#for key in characters_vectors:
#    print('{}\t{}'.format(key, len(characters_vectors[key])))

reciprocal_ranks=[]
ranks=[]
evaluation_file=open('{}/evaluation_results_{}.txt'.format(folder, number),'w')
similarities_file=open('{}/similarities_results_{}.txt'.format(folder, number), 'w')

disappearing_characters=[]

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
                simil=_cosine_similarity(norm_good_vector, norm_other_vector)
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

evaluation_file.write('MRR:\t{}\nMedian rank:\t{}\nMean rank:\t{}\nTotal number of characters considered:\t{}\nCharacters disappearing when dividing the novel in two:\t{}'.format(MRR, median_rank, mean_rank, len(sorted_simil_list)+1, len(char_list)-(len(sorted_simil_list)+1))) 
#results.write('{}\t{}'.format(MRR, folder))
