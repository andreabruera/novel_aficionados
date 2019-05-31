import os
import numpy
import shutil
from distutils.dir_util import copy_tree

hundred_books = os.listdir('dataset/novels/hundred_novels')
steps = numpy.arange(0, len(hundred_books), 3)

for step_index, step in enumerate(steps):
    for book_index, book_dir in enumerate(hundred_books):
        if book_index < step and book_index >= steps[step_index-1]:
            current_folder = 'hundred_novels_by_3_wiki/{}'.format(step_index)
            os.makedirs(current_folder, exist_ok=True)
            copy_tree('dataset/novels/hundred_novels/{}'.format(book_dir), 'dataset/novels/{}/{}'.format(current_folder, book_dir))
