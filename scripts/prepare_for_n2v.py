import nltk
import sys
import utilities

from nltk import sent_tokenize as tok
from utilities import *

char_list=get_characters_list(sys.argv[1])

filename='{}_clean.txt'.format(sys.argv[1])
files=['{}'.format(filename),'{}_part_a'.format(filename),'{}_part_b'.format(filename)]

for i in files:
    f=open('novels/{}'.format(i)).read()
    out=open('novels/{}_ready'.format(i),'w')

    lines=tok(f)

    for line in lines: 
        for alias in char_list:
            if type(alias)==list:
                character=alias[0]
                aliases=alias[1:]
                for name in aliases:
                    line_coref=line.replace(str(name),str(character))
            else:
                line_coref=line
        for i,v in enumerate(line_coref): #character-level
            p=line_coref[i-1]
            if i<len(line_coref)-1:
                n=line_coref[i+1]
            if v.isalpha()==1:
                out.write(v.lower()) 
            else:
                if v==' ':
                    if i!=0:
                        out.write(v)
                    else:
                        pass
                elif i<len(line_coref)-1:
                    if p==' ' or n==' ':
                        pass
                    else:
                        out.write(' ')
                elif i==len(line_coref)-1:
                        if p==' ':
                            pass
                        else:
                            out.write(' ')
        out.write('\n')
                  

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
