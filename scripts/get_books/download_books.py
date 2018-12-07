import requests
import os
import re

filename=open('lists/books_list.txt').readlines()
for l in filename:
    line=l.strip('\n').split('\t')
    b=requests.get('http://www.gutenberg.org/cache/epub/{}/pg{}.txt'.format(line[0], line[0])).text
    name=line[1].replace(' ','_')
    name2=''.join([l for i,l in enumerate(name) if l.isalpha() and i<40 or l=='_' and i<40])
    os.makedirs('100_books/{}'.format(name2), exist_ok=True)
    out=open('100_books/{}/{}.txt'.format(name2, line[0]),'w')
    out.write(b)
