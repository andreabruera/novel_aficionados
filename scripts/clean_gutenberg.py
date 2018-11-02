import sys

file=sys.argv[1]
f=open('book_nlp/data/originalTexts/%s.txt'%(file)).readlines()
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
            c=i.strip('\n').strip(' ').replace('_',' ').replace('*','').replace('--',', ')
            if c!='':
                clean_list.append(c)
            else:
                pass
        else:
            print(i)
            break
clean_book=' '.join(clean_list)
clean_txt=open('book_nlp/data/originalTexts/%s_clean.txt'%(file),'w')
for i in clean_book:
    clean_txt.write(i)
clean_txt.close()
