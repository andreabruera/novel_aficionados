'''First, we get the list of the characters for each novel.
We follow Uncle Podger's Heuristics: we won't consider characters appearing less than 10 times.
'''
def get_characters_list(novel):
    char_list=[]
    with open('novels/{}/{}_characters.txt'.format(novel,novel)) as c:
        char_file=c.readlines()
        for l in char_file:
            l2=l.strip('\n').split('\t')
            if int(l2[0])>=10:
                if '_' in l2[1]:
                    pre_c=l2[1].split('_')
                    c=pre_c[0]
                else:
                    c=l2[1]
                char_list.append(c)
            else:
                break
    return char_list

'''Then, we want to get a list of the three versions of each book (full, part a, part b)'''
def get_books_list(novel):
    books_list=['{}_clean.txt_ready'.format(novel),'{}_clean.txt_part_a_ready'.format(novel),'{}_clean.txt_part_b_ready'.format(novel)]
    return books_list

'''Starting from the list of the books, this function opens each book, and returns a list of sentences'''
def get_novel_vocab(version):
    words=[]
    with open('novels/{}'.format(version)) as book:
        book_list=book.readlines()
        for line in book_list:
            line_words=line.strip('\n').split(' ')
            for w in line_words:
                if w!='':
                    words.append(w)
        yield [words]

'''This is just a small function for opening the books and yielding one line at a time, for training'''
def get_novel_sentences(version,w2v_vocab,character):
    book_final=[]
    vocab_final=[]
    with open('novels/{}'.format(version)) as book:
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