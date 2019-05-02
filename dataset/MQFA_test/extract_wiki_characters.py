import requests
import re
import bs4
import argparse
import nltk
import os
import nonce2vec.utils

from nonce2vec.utils.novels_utilities import * 
from re import sub
from nltk import sent_tokenize
from bs4 import BeautifulSoup as bs

parser=argparse.ArgumentParser()
parser.add_argument("path_to_folder")
parser.add_argument("--number")
args=parser.parse_args()

number=args.number

folder=os.listdir(args.path_to_folder)
for f in folder:
    if '.url' in f:
        url=open('{}/{}'.format(args.path_to_folder, f)).read()
        url=url.strip('\n')
if args.number:
    os.makedirs('{}/quality_test/original_text'.format(args.path_to_folder),exist_ok=True)

    output_text=open('{}/quality_test/original_text/{}.txt'.format(args.path_to_folder, number), 'w')
    

html=requests.get(url).text
wiki_soup=bs(html, features="html5lib")

text=wiki_soup.get_text()
text=text.split('\n')
darcy=[line for line in text]

darcy_final=[]
for section in darcy:
    tok=sent_tokenize(section)
    for sentence in tok:
        sentence=sentence.lower()
        if 'wiki'  not in sentence and 'this article' not in sentence and 'parser' not in sentence and 'js' not in sentence and 'disambiguation' not in sentence:
            sentence=sub('[\W\s0-9]+',' ',sentence)
            sentence=sub('\s\w\s',' ',sentence)
            sentence=sentence.strip(' ')
            if len(sentence)>30:
                darcy_final.append(sentence)

#print(darcy_final)
if args.number:
    for wiki_line in darcy_final:
        output_text.write('{}\n'.format(wiki_line))
#print([line for line in tok if 'Darcy' in line])


#print([line for line in wiki_soup.find_all('a') if 'Darcy' in line])

