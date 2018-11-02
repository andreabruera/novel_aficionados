import re
import sys
from re import sub

file=sys.argv[1]
f=open('book_nlp/data/output/%s_clean/book.id.html'%(file)).read().replace('\t',' ').split('<br />')
for i in f:
    if '<h1>Text' in i:
        break
    else: 
        i2=sub('.*Characters</h1>','',i)
        i3=i2.replace('-- ','')
        i4=sub('\([0-9]*\)','_',i3)
        i5=i4.replace(' _ ','_').strip('_')
        print(i5)

