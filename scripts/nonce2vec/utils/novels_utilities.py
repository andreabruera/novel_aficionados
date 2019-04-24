import gensim
import collections
import nltk
import sys
import nonce2vec
import re

from collections import defaultdict
from nltk import sent_tokenize as tok
from nonce2vec.utils.novels_utilities import *
from re import sub
from gensim.models import Word2Vec

'''First, we get the list of the characters for each novel.
We follow Uncle Podger's Heuristic: we won't consider characters appearing less than 10 times.
'''
def get_characters_list(folder, number):
    char_list=[]
    folder=folder.replace('/processed_novel', '').replace('/temp', '')
    with open('{}/characters_{}.txt'.format(folder, number)) as c:
        char_file=c.readlines()
        for l in char_file:
            l2=l.strip('\n').split('\t')
            if int(l2[0])>=10:
                if '_' in l2[1]:
                    character=l2[1].split('_')
                else:
                    character=l2[1]
                char_list.append(character)
            else:
                break
    return char_list

'''This function goes through the list of the characters and checks their gender, then writing it down to file'''

def get_characters_gender(folder, number, char_list):
    with open('{}/character_gender_{}.txt'.format(folder, number),'w') as g:
        with open('{}/book_nlp_output/{}.tokens'.format(folder, number)) as t:
            t=t.readlines()[1:]
            char_placeholders=defaultdict(int)
            genders_dict=defaultdict(str)
            for line in t:
                if len(char_placeholders.keys())==len(char_list):
                    break
                else:
                    line=line.strip('\n').split('\t')
                    word=line[9].strip('\t')
                    number=line[14].strip('\t')
                    for character in char_list:
                        if type(character)!=list:
                            if 'mr ' in character or 'herr ' in character or 'don' in character:
                                genders_dict[character]='MALE' 
                            elif 'mrs ' in character or 'ms ' in character or 'miss ' in character or 'lady ' in character:
                                genders_dict[character]='FEMALE' 
                            elif character in word.lower() and character not in char_placeholders.keys() and number!='-1':
                                char_placeholders[character]=number
                        elif type(character)==list:
                            unique_alias=character[0]
                            if unique_alias not in char_placeholders.keys() and number!='-1':
                                for alias in character:
                                    if 'mr ' in alias or 'herr ' in alias or 'don' in alias:
                                        genders_dict[unique_alias]='MALE' 
                                    elif 'mrs ' in alias or 'ms ' in alias or 'miss ' in alias or 'lady ' in alias:
                                        genders_dict[unique_alias]='FEMALE' 
                                    elif alias in word.lower():
                                        char_placeholders[unique_alias]=number

            gender_counter=defaultdict(int)
            for char, placeholder in char_placeholders.items():
                gender_counter[placeholder]=[0,0]
            female_pronouns=['she', 'her']
            male_pronouns=['he', 'him', 'his']
            for line in t:
                line=line.strip('\n').split('\t')
                word=line[9].strip('\t')
                number=line[14].strip('\t')
                for placeholder in gender_counter.keys():
                    if int(placeholder)==int(number) and 'NNP' not in line[10].strip('\t'):
                        if word.lower()=='she' or word.lower()=='her':
                            gender_counter[placeholder][0]+=1
                        elif word.lower()=='he' or word.lower()=='his' or word.lower()=='him':
                            gender_counter[placeholder][1]+=1

            for placeholder, counters in gender_counter.items():
                for name, number in char_placeholders.items():
                    if number==placeholder:
                        character=name
                if counters[0]>counters[1]:
                    genders_dict[character]='FEMALE'
                elif counters[0]<counters[1]:
                    genders_dict[character]='MALE'
                else:
                    genders_dict[character]='UNKNOWN'

            for char in char_list:
                if type(char)==list:
                   character=char[0]
                else:
                   character=char
                g.write('{}\t{}\n'.format(character, genders_dict[character]))

    return genders_dict

'''Then, we want to get a list of the two versions of each book (part a, part b)'''
def get_books_dict(number):
    versions={}
    parts=['a','b']
    for i in parts:
        versions[i]='{}_part_{}_n2v'.format(number, i)
    return versions

'''Starting from the list of the books, this function opens each book, and returns a list of sentences'''
def get_novel_vocab(folder, filename):
    words=[]
    with open('{}/{}'.format(folder, filename)) as book:
        book_list=book.readlines()
        for line in book_list:
            line_words=line.strip('\n').split(' ')
            for w in line_words:
                if w!='':
                    words.append(w)
        yield [words]

'''This is just a small function for opening the books and yielding one line at a time, for training'''
def get_novel_sentences(folder, filename, w2v_vocab, character):
    book_final=[]
    vocab_final=[]
    with open('{}/{}'.format(folder, filename)) as book:
        book_list=book.readlines()
        for line in book_list:
            line_list=[]
            vocab_list=[]
            line_words=line.strip('\n').split(' ')
            for w in line_words:
                if w=='' or w not in w2v_vocab:
                    if w!=character:
                        pass
                    else:
                        line_list.append('___')
                        vocab_list.append(character)
                else:
                    if w==character:
                        line_list.append('___')
                        vocab_list.append(character)
                    else: 
                        line_list.append(w)
                        vocab_list.append(w)
            book_final.append(line_list)
            vocab_final.append(vocab_list)
        return book_final, vocab_final

def prepare_for_n2v(folder, number, filename, w2v_model=None, write_to_file=False, wiki_novel=False):

    ####### STEP 1: setting up the folders' path variables
    
    book_nlp_output_folder='{}/book_nlp_output'.format(folder)
    temp_folder='{}/temp'.format(folder)
    processed_novel_folder='{}/processed_novel'.format(folder)

    if wiki_novel==False:
        filename=filename.replace('_clean.txt_','_')
    else:
        filename='{}/quality_test/original_text/{}.txt'.format(folder, number)

    #######  STEP 2: from the output of booknlp to a file containing the list of characters and the number of times they occur

    f=open('{}/book.id.html'.format(book_nlp_output_folder)).read().split('<br />')
    out=open('{}/characters_{}.txt'.format(folder, number),'w')
    for i in f:
        if '<h1>Text' in i:
            break
        else:
            i2=sub('.*Characters</h1>','',i)
            i3=i2.replace('-- ','')
            i4=sub('\([0-9]*\)','_',i3)
            i5=i4.replace(' _ ','_').strip('_')
            i6=i5.replace('\t ','\t')
            i7=sub(r'[^\w\s]','',i6)
            if 'Gutenberg' not in i7:
                out.write('{}\n'.format(i7.lower()))
    out.close()
    print('Created the list of the characters, and written it to file')

    ########   STEP 3: creating a list of characters from that file, by using a function in nonce2vec.utils.novels_utilities

    char_list=get_characters_list(folder, number)

    genders_dict=get_characters_gender(folder, number, char_list)

    ########   STEP 4: loading the model created with Word2Vec, in order to access its vocabulary
    if wiki_novel==False:
        print('Loading the background model for checking the vocabulary')
        background_model=Word2Vec.load(w2v_model)
        print('Creating the background model\'s vocabulary')
        background_vocab=[k for k in background_model.wv.vocab]

    ########    STEP 5: creating the final version of the txt to be used for training on N2V. Main features are 1) one sentence per line, 2) different names for the same character are substituted with one single name, 3) punctuation is removed and double/triple backspaces, common within Gutenberg files, are removed
    
    #files=['{}'.format(filename),'{}_part_a'.format(filename),'{}_part_b'.format(filename)]

    current_char_list=[]
    full_novel=[]
    print('Creating the final version of novel {} for version: {}'.format(number, filename))
    if wiki_novel==False:
        f=open('{}'.format(filename)).read()
        out=open('{}_n2v'.format(filename),'w')
        lines=tok(f)
    else:
        lines=open('{}'.format(filename)).readlines()
        out=open('{}_n2v'.format(filename),'w')

    
    print('Trimming the novels, so as to take away all new words except the characters...')
    
    for line in lines:
        if 'chapter' not in line and 'gutenberg' not in line and 'email' not in line and 'copyright' not in line: 

            line=line.strip('\n').lower()
            line=sub(r'[^\w\s]', '', line)

            for alias in char_list:

                if type(alias)==list:

                    first_name=alias[0]

                    if ' ' in first_name:
                        first_name=first_name.replace(' ', '_')

                    if first_name not in current_char_list:
                        current_char_list.append(first_name)

                    for a in alias:
                        if ' {} '.format(a) in line:
                            line=line.replace(' {} '.format(a), ' {} '.format(first_name))

                else:

                    first_name=alias.replace(' ', '_')
                    if first_name not in current_char_list:
                        current_char_list.append(first_name)

                    if alias in line:

                        if ' ' in alias:
                            line=line.replace(' {} '.format(alias), ' {} '.format(first_name))

            line2=re.sub(r'\W+',r' ',line)
            line3=line2.strip(' ')
            words=line3.split(' ')
            line4=[]
            for w in words:
                word_lowercase=w.lower()
                line4.append(word_lowercase)
            full_novel.append(line4)
            if write_to_file==True:
                line5=' '.join(line4)
                out.write('{}\n'.format(line5))

    print('Characters\' names reduced to the following list: {}'.format(current_char_list))


    if wiki_novel==False:
        novel_versions={}
        mid_novel=int(len(full_novel)/2)

        novel_versions['{}_part_a'.format(filename)]=full_novel[:mid_novel]
        novel_versions['{}_part_b'.format(filename)]=full_novel[mid_novel:]
        print('Double versions done!'.format(i))
        return novel_versions, current_char_list, background_vocab, genders_dict 
    elif wiki_novel==True:
        novel_versions=full_novel
        return novel_versions, current_char_list, genders_dict 

### NOVELS NOTE: this function is the alternative version to get the novels' sentences without having saved the two version of each novel to file (which, admittedly, is fairly useless).

def get_novel_sentences_from_versions_dict(version, character, background_vocab=None):
    book_final=[]
    vocab_final=[]
    for line in version:
        line_list=[]
        vocab_list=[]
        for w in line:
            if w=='':
                pass
            else:
                if w==character:
                    line_list.append('___')
                    vocab_list.append(character)
                else: 
                    line_list.append(w)
                    vocab_list.append(w)
        book_final.append(line_list)
        vocab_final.append(vocab_list)
    return book_final, vocab_final
