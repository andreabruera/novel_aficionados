import utilities
import numpy 
import math
import sys

from utilities import *
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

def extract_char_vectors(book):
    novels_list=utilities.get_books_list(book)
    char_list=utilities.get_characters_list(book)

    books_vectors=['{}.character_vectors'.format(i) for i in novels_list]
    vectors_dict={}
    for version in books_vectors:
            for index, character2 in enumerate(char_list):
                character=character2.replace(' ','_')
                vectors_dict['{}_{}'.format(character, version)]=[]
                f=open('data/{}'.format(version)).readlines()
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

characters_vectors,index=extract_char_vectors('308')

for key in characters_vectors:
    print('{}\t{}'.format(key, len(characters_vectors[key])))

def _cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = numpy.dot(peer_v, query_v)
    den_a = numpy.dot(peer_v, peer_v)
    den_b = numpy.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


results=open('results/results.txt','w')
#results_unsorted=open('results_unsorted.txt','w')
for key in characters_vectors:
    good_vector=numpy.array(characters_vectors[key],dtype=float)
    results.write('\nResults for the vector: {}\n\n'.format(key))
    #results_unsorted.write('\nResults for the vector: {}\n\n'.format(key))
    simil_dict={}
    for other_key in characters_vectors:
        other_vector=numpy.array(characters_vectors[other_key],dtype=float)
        simil=_cosine_similarity(good_vector,other_vector)
        simil_dict[float(simil)]=other_key
    sorted_simil_dict=sorted(simil_dict,reverse=True)
    for s in sorted_simil_dict:    
    #for s in simil_dict:    
        results.write('\t{} - {}\n'.format(s,simil_dict[s].replace('_308_clean.txt_','_').replace('_ready.character_vectors','')))
        #results_unsorted.write('\t{} - similarity: {}\n'.format(s,simil_dict[s]))

