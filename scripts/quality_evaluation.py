import os
import pickle
import argparse
import re
import nonce2vec 
import numpy 
import math
import sys

from nonce2vec.utils.novels_utilities import *
from numpy import dot,sqrt,sum,linalg
from math import sqrt

parser=argparse.ArgumentParser()
parser.add_argument('--training_folder', type=str, required = True)
args=parser.parse_args()

output_folders=[names for names in os.listdir(args.training_folder)]

for setup_folder in output_folders:

    current_folder='{}/{}'.format(args.training_folder, setup_folder)
    data_output_folder='{}/data_output/MQFA_test'.format(current_folder)
    
    #novels_folders=[names for names in os.listdir(current_folder)]

    wiki=[]
    for root, name, files in os.walk(current_folder):
        for f in files:
            if 'pickle' in f:
                wiki.append(os.path.join(root,f))

    n2v_folders=[names for names in os.listdir('plot_folders')]

    for n2v_folder in n2v_folders:
        if 'sum' in n2v_folder:
            model_name='sum'
        else:
            model_name='n2v'
        numbers=[] 
        current_setup=[]
        for root, name, files in os.walk('plot_folders/{}'.format(n2v_folder)):
            for f in files:
                if 'pickle' in f and 'Brothers' not in root:
                    numbers.append(f.replace('.pickle',''))
                    current_setup.append(os.path.join(root,f))

        output_folders_dict={}
        paths_dict={}
        for number in numbers:
            paths_dict[number]=[]
            for p in wiki:
                if 'wiki_{}.pickle'.format(number) in p:
                    paths_dict[number].append(p)
            for p in current_setup:
                if '{}.pickle'.format(number) in p:
                    paths_dict[number].append(p)

        for index, number_key in enumerate(paths_dict.keys()):
            reciprocal_ranks=[]
            ranks=[]

            evaluation_file=open('quality_test_novels/{}/{}/quality_{}_evaluation_results_{}.txt'.format(setup_folder, novels_folders[index], model_name, number_key),'w')
            similarities_file=open('quality_test_novels/{}/{}/quality_{}_similarities_results_{}.txt'.format(setup_folder, novels_folders[index], model_name, number_key), 'w')
            
            final_dict_a={}
            final_dict_b={}
            final_char_dict={}
            evaluation_dict={}
            for novel in paths_dict[number_key]:
                pickled_dict=pickle.load(open(novel,'rb'))
                for key in pickled_dict.keys():
                    evaluation_dict[key]=pickled_dict[key]
                    clean_key=re.sub(r'_a$|_b$','',key)
                    if clean_key not in final_char_dict.keys():
                        final_char_dict[clean_key]=1
                    else:
                        final_char_dict[clean_key]+=1
            for alias in final_char_dict.keys():
                if final_char_dict[alias]>2 and alias in evaluation_dict.keys():
                    final_dict_a[alias]=evaluation_dict[alias]
                    final_dict_b[alias]=evaluation_dict[alias]
                    final_dict_a['{}_a'.format(alias)]=evaluation_dict['{}_a'.format(alias)]
                    final_dict_b['{}_b'.format(alias)]=evaluation_dict['{}_b'.format(alias)]
            
            for name, vector in final_dict_a.items():
                vector=normalise(vector)
                final_dict=final_dict_a
                similarities={}
                for other_name, other_vector in final_dict.items():
                    if '_b' in other_name or '_a' in other_name and name!='tinker_bell':
                        other_vector=normalise(other_vector)
                        if other_name != name:
                            similarities[float(_cosine_similarity(vector, other_vector))]=other_name
                    sorted_simil_list=sorted(similarities, reverse=True)

                for rank, similarity in enumerate(sorted_simil_list):
                    rank+=1
                    current_character=similarities[similarity].split('_')
                    if current_character[0] == name:
                        reciprocal_ranks.append(1/rank)
                        ranks.append(rank)
                        #print('\nResult for the vector: {}\nRank: {} out of {} characters\nReciprocal rank: {}\nCosine similarity to the query: {}\n'.format(name, rank, len(sorted_simil_list), (1/rank), similarity))

                        #for index, i in enumerate(sorted_simil_list):
                        #    print('Position: {} - {} - {}'.format(index+1, similarities[i], i))
                        #    pass
                        similarities_file.write('\nResult for the vector: {}\nRank: {} out of {} characters\nReciprocal rank: {}\nCosine similarity to the query: {}\n'.format(name, rank, len(sorted_simil_list), (1/rank), similarity))
                        for i in sorted_simil_list:
                            similarities_file.write('{} - {}\n'.format(similarities[i], i))

            MRR=numpy.mean(reciprocal_ranks)
            median_rank=numpy.median(ranks)
            mean_rank=numpy.mean(ranks)
            #print(similarities)


            #print('Novel:\t{}\n\nMRR:\t{}\nMedian rank:\t{}\nMean rank:\t{}\nTotal number of characters considered:\t{}\n\n'.format(novels_folders[index], MRR, median_rank, mean_rank, len(final_char_dict.keys()))) 

            evaluation_file.write('Novel":\t{}\nMRR:\t{}\nMedian rank:\t{}\nMean rank:\t{}\nTotal number of characters considered:\t{}\n\n'.format(novels_folders[index], MRR, median_rank, mean_rank, len(final_char_dict.keys()))) 
