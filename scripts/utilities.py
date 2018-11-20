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
                    c=l2[1].split('_')
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
