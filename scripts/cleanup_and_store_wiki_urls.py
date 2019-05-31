import requests
import re
import bs4
import argparse
import nltk
import os
import nonce2vec.utils
import sys

from nonce2vec.utils.novels_utilities import * 
from re import sub
from nltk import sent_tokenize
from bs4 import BeautifulSoup as bs

parser=argparse.ArgumentParser()
parser.add_argument("--novels_folder", type=str, required=True, help='Folder where the novels are stored')
args=parser.parse_args()

folder=os.walk(args.novels_folder)

for root, directory, files in folder:
    for current_file in files:
        if '.url' in current_file:
            print(root)
            url=open('{}'.format(os.path.join(root, current_file))).read()
            url=url.strip('\n')
            os.makedirs('{}/original_wikipedia_page'.format(root),exist_ok=True)
            #print('{}/original_wikipedia_page'.format(root))
            number = re.sub('.txt', '', os.listdir('{}/original_novel'.format(root))[0])
            print(number)

            html=requests.get(url).text
            wiki_soup=bs(html, features="html5lib")

            
            text=wiki_soup.get_text()

            cleaned_up_text = []

            text=text.split('\n')
            whole_text=[re.sub('\s+', ' ',line) for line in text if line != '']
            marker = False
            for line in whole_text:
                if line == 'References[edit]':
                    break
                elif line == ' Jump to search':
                    marker = True
                if marker==True and line != ' Jump to search':
                    tok=sent_tokenize(line)
                    cleaned_up_text.append(tok)
            whole_text_semi_final = [sentence for group in cleaned_up_text for sentence in group][1:]

            whole_text_final = []
            for sentence in whole_text_semi_final:
                sentence = re.sub('\[.+\]|^\W+|\\\'s|\(\d+\)|^\s|$\s', '',sentence)
                sentence = re.sub('\d+s|\d+th|\d+nd|\d+st|\d+rd','', sentence)
                sentence = re.sub('\d','', sentence)
                sentence = re.sub('\s\W+\s|\s\w\s',' ', sentence)
                sentence = re.sub('-',' ', sentence)
                sentence = re.sub('\'s',' ', sentence)
                #sentence = re.sub('\W+',' ', sentence)
                sentence = re.sub('\s+',' ', sentence)
                sentence = re.sub('^\W+|\s+$','', sentence)
                if sentence != '' and len(sentence.split())>5 and 'disambiguation' not in sentence and 'Wiki' not in sentence and 'parser' not in sentence and '{' not in sentence:
                    whole_text_final.append(sentence)
            with open('{}/original_wikipedia_page/{}.txt'.format(root, number), 'w') as output_text:
                for sentence_index, sentence in enumerate(whole_text_final):
                    if sentence_index == 0 and 'Author' in sentence:
                        pass
                    else:
                        output_text.write('{}\n'.format(sentence))
