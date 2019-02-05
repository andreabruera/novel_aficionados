import os

cwd=os.getcwd()
big='{}/problemi_big_test_novels'.format(cwd)

for setup in os.listdir(big):
    setup_folder=os.listdir('{}/{}'.format(big, setup))
    mrr=0.0
    median=0.0
    characters=0
    counter=0
    for novel in setup_folder:
        novel_folder=os.listdir('{}/{}/{}'.format(big, setup, novel))
        for single_file in novel_folder:
            if 'evaluation' in single_file:
                evaluation=open('{}/{}/{}/{}'.format(big, setup, novel, single_file)).readlines()
                line1=evaluation[0].strip('\n').split('\t')[1]
                line2=evaluation[1].strip('\n').split('\t')[1]
                line3=evaluation[2].strip('\n').split('\t')[1]
                mrr+=float(line1)
                median+=float(line2)
                characters+=int(line3)
                counter+=1
    average_mrr=mrr/float(counter)
    average_median=median/float(counter)
    average_characters=characters/counter
    print('Setup: {}\n\nAverage MRR: {}\nAverage Median: {}\nAverage number of characters: {}'.format(setup, average_mrr, average_median, average_characters))
