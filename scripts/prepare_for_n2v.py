import nltk
import sys
import nonce2vec 
import re

from nltk import sent_tokenize as tok
from nonce2vec.utils.novels_utilities import *

folder=sys.argv[1]
number=sys.argv[2]
filename=sys.argv[3]

char_list=get_characters_list(folder, number)

files=['{}'.format(filename),'{}_part_a'.format(filename),'{}_part_b'.format(filename)]

for i in files:
    f=open('{}'.format(i)).read()
    out_filename=i.replace('_clean.txt_','')
    out=open('{}_n2v'.format(out_filename),'w')

    lines=tok(f)

    for line in lines:
        line=line.strip('\n').strip('\b')
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
'''        for i,v in enumerate(line): #character-level
            if v.isalpha()==True:
                out.write(v.lower()) 
            elif v.isalpha()==False:
                if v==' ':
                    if i>0 and i<len(line)-1:
                        previous_letter=line[i-1]
                        if previous_letter.isalpha()==True:
                            out.write(' ')
                        elif previous_letter==' ':
                            pass
                        elif previous_letter.isalpha()==False:
                            if i>1:
                                double_previous_letter=line[i-2]
                                if double_previous_letter!=' ':
                                    out.write(' ')
                                else:
                                    pass
                    else:
                        pass
                else:
                    if i>0 and i<len(line)-1:
                        next_letter=line[i+1]
                        previous_letter=line[i-1]
                        if previous_letter.isalpha()==True and next_letter.isalpha()==True:
                            out.write(' ')
                        else:
                            pass
                    else:
                        pass      
        out.write('\n')
                  
'''
'''
            c=i.strip(' ').\
lstrip('\n').\
replace('_',' ').\
replace('\'','').\
replace('*','').\
replace('--',' ').\
replace(',','').\
replace('"','').\
replace('?','').\
replace('!','').\
replace(';','').\
replace('\'s','').\
replace('(','').\
replace(')','').\
replace('-',' ').\
replace('{','').\
replace('}','').\
replace(':','').\
replace('&',' ').\
replace('[','').\
replace(']','').\
strip('\n').\
strip('\r').\
replace('   ',' ').\
replace('  ',' ').\
replace('mr.','').\
replace('mrs.','').\
replace('ms.','')

            if c!='':
'''
