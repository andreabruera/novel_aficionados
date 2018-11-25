import utilities
import sys

from utilities import *

novels_list=utilities.get_books_list(sys.argv[1])
char_list=utilities.get_characters_list(sys.argv[1])

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
                                    vectors_dict['{}_{}'.format(character, version)].append(w.replace('[','').replace(']',''))
                        elif marker==True and f2[0]==char_list[index+1].replace(' ','_'):
                            marker=False
                            break
                    else:
                        for w in f3:
                            if w=='[' or w==']' or len(w)==0:
                                pass
                            else:
                                if len(w)>0:
                                    vectors_dict['{}_{}'.format(character, version)].append(w.replace(']','').replace('[',''))
                elif marker==False and f2[0]==character:
                    marker=True
                    for w in f3:
                        if w=='[':
                            pass
                        else:
                            if len(w)>0:
                                vectors_dict['{}_{}'.format(character, version)].append(w.replace('[','').replace(']',''))
for key in vectors_dict:
    print('{}\t{}'.format(key, len(vectors_dict[key])))
