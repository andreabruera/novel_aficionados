import nltk
import sys
import nonce2vec 
import re

from nltk import sent_tokenize as tok
from nonce2vec.utils.novels_utilities import *
from re import sub

#folder=sys.argv[1]
#number=sys.argv[2]
#filename=sys.argv[3]

def prepare_for_n2v(folder, number, filename, w2v_model):

    #######  STEP 1: from the output of booknlp to a file containing the list of characters and the number of times they occur

    f=open('{}/book_nlp_output/book.id.html'.format(folder)).read().split('<br />')
    out=open('{}/characters_{}.txt'.format(folder, number),'w')
    for i in f:
        if '<h1>Text' in i:
            break
        else:
            i2=sub('.*Characters</h1>','',i)
            i3=i2.replace('-- ','')
            i4=sub('\([0-9]*\)','_',i3)
            i5=i4.replace(' _ ','_').strip('_')
            i6=i5.replace('\t ','\t')
            i7=sub(r'[^\w\s]','',i6)
            if 'Gutenberg' not in i7:
                out.write('{}\n'.format(i7.lower()))
    out.close()

    ########   STEP 2: creating a list of characters from that file, by using a function in nonce2vec.utils.novels_utilities

    char_list=get_characters_list(folder, number)
    
    gender_list=get_characters_gender(folder, number, char_list)

    print(gender_list)

    ########    STEP 3: creating the final version of the txt to be used for training on N2V. Main features are 1) one sentence per line, 2) different names for the same character are substituted with one single name, 3) punctuation is removed and double/triple backspaces, common within Gutenberg files, are removed

    files=['{}'.format(filename),'{}_part_a'.format(filename),'{}_part_b'.format(filename)]
    add_to_char_list=1
    char_dict={}
    for i in files:
        char_dict_part={}
    #    f=open('../{}'.format(i)).read()
        f=open('{}'.format(i)).read()

        out_filename=i.replace('_clean.txt_','_').replace('/temp','/processed_novel')
    #    out=open('../{}_n2v'.format(out_filename),'w')
        out=open('{}_n2v'.format(out_filename),'w')

        lines=tok(f)

        for line in lines:
            line=line.strip('\n')
            for alias in char_list:
                if type(alias)==list:
                    for a in alias:
                        a=sub(r'\W+', ' ', a)
                    first_name=alias[0]
                    if ' ' in first_name:
                        name_parts=first_name.split(' ')
                        character=name_parts[1]
                    else:
                        character=first_name
                    aliases=alias[1:]
                    for name in aliases:
                        if name in line:
                            char_dict_part[character]+=1
                            line=line.replace(str(name), str(character)) 
                else:
                    if alias in line:
                        alias=sub(r'\W+', ' ', alias)
                        if ' ' in alias:
                            name_parts=alias.split(' ')
                            character=name_parts[1]
                        line=line.replace(alias, character)
                        char_dict_part[character]+=1
                    else:
                        pass

            line2=re.sub(r'\W+',r' ',line)
            line3=line2.strip(' ')
            out.write('{}\n'.format(line3.lower())) 
        add_to_char_list_final==False
