import sys
import nltk
import os

from nltk import sent_tokenize as tok

folder=sys.argv[1]
base_folder=folder.replace('/original_novel','')
temp_folder='{}/temp'.format(base_folder)
booknlp_folder='{}/book_nlp_output'.format(base_folder)
processed_novel_folder='{}/processed_novel'.format(base_folder)
os.makedirs('{}'.format(temp_folder), exist_ok=True)
os.makedirs('{}'.format(booknlp_folder), exist_ok=True)
os.makedirs('{}'.format(processed_novel_folder), exist_ok=True)

number=sys.argv[2]

f=open('{}/{}.txt'.format(folder, number)).readlines()
out_clean=open('{}/{}_clean.txt'.format(temp_folder, number),'w')
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

