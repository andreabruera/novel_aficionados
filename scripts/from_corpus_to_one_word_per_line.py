### EXAMPLE: python3 -m scripts.from_corpus_to_one_word_per_line ../19337_clean.txt_n2v ../scrooge_one_word.txt19337_clean.txt_n2v ../scrooge_one_word.txt

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('corpus', type = str, help = 'Path to the file to be turned into a one-word-per-line file')
parser.add_argument('output_name', type = str, help = 'Path to the output file')

args = parser.parse_args()

with open(args.corpus) as f:
    out=open(args.output_name, 'w')
    for line in f:
        line=re.sub('\d+', ' ', line)
        #line=re.sub('\s+|\W+', ' ', line)
        line=re.sub('\s+', ' ', line)
        line=line.split()
        for word in line:
            if word != '' and word.isalpha():
                out.write('{}\n'.format(word))
