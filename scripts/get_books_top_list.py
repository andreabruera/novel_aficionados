import bs4
import sys
import requests
import os
import re

from bs4 import BeautifulSoup as bs

top_page=str(sys.argv[1])
file=bs(open('../pages/{}'.format(top_page)),features="html5lib")
file_tags=file.find_all('a')
'''
os.makedirs('../lists',exist_ok=True)
authors_list=open('../lists/authors_list.txt','w')
os.makedirs('../lists',exist_ok=True)
books_list=open('../lists/books_list.txt','w')
'''
books_dict={}
authors_dict={}
for i in file_tags:
    link=i['href']
    text=i.text
    if 'ebooks/' in link and '(' in text:
        text_clean=re.sub('\(.*\)','',text)
        l2=re.sub('\/.*\/','',link)
        if l2 not in books_dict.keys():
#            books_list.write('{}\n'.format(text_clean))
            books_dict[l2]=text_clean
    elif 'authors/' in link:
        if link not in authors_dict.keys():
                author_code=re.sub('\/.*\/.*\/','',link)
                authors_dict[author_code]=text
#                authors_list.write('{}\t{}\n'.format(author_code,text))

books_list=sorted(books_dict)
authors_list=sorted(authors_dict)
websites_dict={}
initials_list=[]

for code in authors_list:
    initial=re.sub('\#.*','',code)
    if initial not in websites_dict.keys(): 
        initials_list.append(initial)
        websites_dict[initial]=[]
        websites_dict[initial].append(code.replace(initial,''))
    else:
        websites_dict[initial].append(code.replace(initial,''))

print(websites_dict.items())

h=requests.get('https://www.gutenberg.org/browse/authors/{}'.format(initials_list[1]))
h_text_first=h.text
h_text=h_text_first.split('\n')
from_authors_to_books=[]
c=0
for code in websites_dict[initials_list[1]]:
    for i,line in enumerate(h_text):
        if '"{}"'.format(code) in line:
            c+=1
            new_lines=h_text[i+1:]
            for l in new_lines:
                if '<ul>' in l:
                    pass
                if '</ul>' in l:
                    break
                else:
                    from_authors_to_books.append(l)
print('total of writers to be found: {}'.format(len(websites_dict[initials_list[1]])))
print('total of writers actually found: {}'.format(c))
print('total of novels actually found: {}'.format(len(from_authors_to_books)))

semi_final_soup=bs('\n'.join(from_authors_to_books),features="html5lib")
readable_semi_final=semi_final_soup.find_all('a')
for a_tag in readable_semi_final:
    number=a_tag['href']
    number_clean=re.sub('\/.*\/','',number)
    text_clean=readable_semi_final[3].text
    if number_clean not in books_dict.keys():
        books_dict[number_clean]=text_clean
print('{}'.format(len(books_dict.keys())))
#i=h_soup.find_all('h2')[78]
#print(i.find_all('a')[1]['href'])

