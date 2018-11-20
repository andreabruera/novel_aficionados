import sys
import nltk

from nltk import sent_tokenize as tok

file=sys.argv[1]
f=open('novels/{}.txt'.format(file)).readlines()
out_clean=open('novels/{}_clean.txt'.format(file),'w')
clean_list=[]
for v,i in enumerate(f):
    if '***START' not in str(i) and v!=len(f)-1:
        pass
    elif v==len(f)-1:
        malandrino=f
    else:
        malandrino=f[(v+1):]
        break
for i in malandrino:
    if '***END' not in str(i):
        c=i.replace('_',' ').strip(' ').strip('\n').replace('\r','')
        clean_list.append('{}'.format(c))
    else:
        break 
clean_book=' '.join(clean_list)
sent_book=tok(clean_book)
for i in sent_book:
    out_clean.write('{}\n'.format(i))

out_clean.close()
