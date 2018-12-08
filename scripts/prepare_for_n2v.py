import nltk
import sys
import nonce2vec 
import re

from nltk import sent_tokenize as tok
from nonce2vec.utils.novels_utilities import *
from re import sub

folder=sys.argv[1]
number=sys.argv[2]
filename=sys.argv[3]

#######  STEP 1: from the output of booknlp to a file containing the list of characters and the number of times they occur

f=open('{}/booknlp/book.id.html'.format(folder)).read().split('<br />')
out=open('{}/{}_characters.txt'.format(folder, number),'w')

for i in f:
    if '<h1>Text' in i:
        break
    else:
        i2=sub('.*Characters</h1>','',i)
        i3=i2.replace('-- ','')
        i4=sub('\([0-9]*\)','_',i3)
        i5=i4.replace(' _ ','_').strip('_')
        i6=i5.replace('\t ','\t')
        if 'Gutenberg' not in i5:
            out.write('%s\n'%(i5.lower()))
out.close()

########   STEP 2: creating a list of characters from that file, by using a function in nonce2vec.utils.novels_utilities

char_list=get_characters_list(folder, number)

########    STEP 3: creating the final version of the txt to be used for training on N2V. Main features are 1) one sentence per line, 2) different names for the same character are substituted with one single name, 3) punctuation is removed and double/triple backspaces, common within Gutenberg files, are removed

files=['{}'.format(filename),'{}_part_a'.format(filename),'{}_part_b'.format(filename)]

for i in files:
    f=open('../{}'.format(i)).read()
    out_filename=i.replace('_clean.txt_','_')
    out=open('../{}_n2v'.format(out_filename),'w')

    lines=tok(f)

    for line in lines:
        line=line.strip('\n')
        for alias in char_list:
            if type(alias)==list:
                character=alias[0]
                aliases=alias[1:]
                for name in aliases:
                    line=line.replace(str(name),str(character))
            else:
                pass 
        line2=re.sub(r'\W+',r' ',line)
        line3=line2.strip(' ')
        out.write('{}\n'.format(line3.lower())) 
