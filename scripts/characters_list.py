import re
import sys
from re import sub

folder=sys.argv[1]
number=sys.argv[2]

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
