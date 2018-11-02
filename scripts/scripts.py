import zipfile,os,string,requests,html5lib,bs4,sys
from bs4 import BeautifulSoup as bs
from zipfile import ZipFile as zf

def shelf_creator(url):
	books=[]
	r=requests.get(url)
	h=r.text
	out=open('shelves/all_shelves.txt','a')
	soup=bs(h,'html5lib')
	for i in soup.find_all('a'):
		i2=str(i)
		if 'title' in i2 and 'org' in i2:
			l=i2.strip('\n')
			l2=l.\
replace('<a class="extiw" href="//www.gutenberg.org/ebooks/','').\
replace('" title="ebook:',' ').\
replace('">',' ').\
replace('</a>',' ').\
replace('<br/>',' ').\
replace("'","").\
replace('.','').\
replace('-',' ').\
replace('_',' ').\
replace(',','').\
replace(';','').\
replace('!','').\
replace('/','').\
replace('?','').\
replace(':','')
			w=l2.split(' ')
			del w[1]
			#out.write(' '.join(w)+'\n')
			books.append(w)
	#print(books)
	return books
'''
def books_numbers_only(file_in,file_out):
	for i in f:
		i2=i.strip('\n')
		l=i2.split(' ')
		if l[0].isdigit():
			out.write(l[0]+'\n'
'''
def books_numbers_with_names(file_in,file_out):
    f=open(str(file_in)).readlines()
    f_out=open(str(file_out),'w')
    for i in f:
        i2=i.strip('\n')
        l=i2.split(' ')
        u=l[0]
        if u.isdigit():
            del l[1]
            line=' '.join(l)
            f_out.write(str(line)+'\n')
        else:
            f_out.write(str(l[0])+'\n')
    f_out.close()
    return f

def unzippa(books):
	ans=input('list or file?')
	if ans == 'file':
		file=input('Path to file?    ')
		books=open(file).readlines()
	if ans == 'list':
		print('using books list')
	count=0
	books_count=0
	print(len(books))
	path=os.getcwd()
	list_dir=os.listdir('%s/books'%path)
	missing_files=[]
	for i in books:
		#i2=str(i).strip('\n')
		#l=i2.split(' ')
		#print(i)
		if i[0].isdigit():
			books_count+=1
			c=i[1:]
			end=c[len(c)-1]
			if end=='':
				del c[len(c)-1]
			name='_'.join(c)
			zip_name='%s.zip'%str(i[0])
			#print('%s %s'%(name,zip_name))
			if zip_name in list_dir:
				count+=1
				try:
					os.mkdir('books/final/%s'%name)
				except(FileExistsError):
					pass
				zip=zf('books/%s'%zip_name)
				#print(zf.printdir(zip))
				zf.extractall(zip,path='books/final/%s'%name)
			else:
				zero='%s-0.zip'%str(i[0])
				eight='%s-8.zip'%str(i[0])
				if zero in list_dir:
					count+=1
					try:
						os.mkdir('books/final/%s'%name)
					except(FileExistsError):
						pass
					zip=zf('books/%s'%zero)
					#print(zf.printdir(zip))
					zf.extractall(zip,path='books/final/%s'%name)
				elif eight in list_dir:
					count+=1
					try:
						os.mkdir('books/final/%s'%name)
					except(FileExistsError):
						pass
					zip=zf('books/%s'%eight)
					#print(zf.printdir(zip))
					zf.extractall(zip,path='books/final/%s'%name)
				else:
					print(zip_name)
					missing_files.append(zip_name)

	print(books_count)
	print(count)
	print(len(missing_files))
	u=open('missing_files.txt','w')
	for l in missing_files:
		u.write('%s\n'%l)	

### START OF THE SCRIPT

u=open('shelves/webpages_list.txt').readlines()
books=[]
for i in u:
	i2=i.strip('\n')
	shelf=shelf_creator(i2)
	for x in shelf:
		books.append(x)
unzippa(books)
