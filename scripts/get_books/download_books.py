import requests
import os
import re
import sys

folder=sys.argv[2]

filename=open('scripts/get_books/lists/books_list.txt').readlines()
#filename=open('scripts/get_books/lists/grid_search_books_list.txt').readlines()
for l in filename:
    line=l.strip('\n').split('\t')
    b=requests.get('http://www.gutenberg.org/cache/epub/{}/pg{}.txt'.format(line[0], line[0])).text
    name=line[1].replace(' ','_')
    name2=''.join([l for i,l in enumerate(name) if l.isalpha() and i<40 or l=='_' and i<40])
    os.makedirs('{}/{}'.format(folder, name2), exist_ok=True)
    os.makedirs('{}/{}/original_novel'.format(folder, name2), exist_ok=True)
    out=open('{}/{}/original_novel/{}.txt'.format(folder, name2, line[0]),'w')
    out.write(b)
