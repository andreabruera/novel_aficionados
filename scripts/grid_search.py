import os
import numpy
import matplotlib.pyplot as plt
import matplotlib
import re
from re import sub

cwd=os.getcwd()
#big='{}/big_test_novels'.format(cwd)
big='{}/plot_folders'.format(cwd)
os.makedirs('plots', exist_ok=True)
plot_median={}
plot_mrr={}
lengths={}
names={}
characters_dict={}
characters_std={}

for setup in os.listdir(big):
    setup_folder=os.listdir('{}/{}'.format(big, setup))
    #median=[]
    #characters=0
    characters=[]
    counter=0
    list_var_mrr=[]
    list_var_median=[]
    list_var_mean=[]

    plot_median[setup]=[]
    plot_mrr[setup]=[]
    lengths[setup]=[]
    names[setup]=[]
    characters_dict[setup]=[]
    characters_std[setup]=[]
    
    for novel in setup_folder:
        characters_frequency=[]
        novel_folder=os.listdir('{}/{}/{}'.format(big, setup, novel))
        for single_file in novel_folder:
            if 'evaluation' in single_file:
                evaluation=open('{}/{}/{}/{}'.format(big, setup, novel, single_file)).readlines()
                line1=evaluation[0].strip('\n').split('\t')[1]
                line2=evaluation[1].strip('\n').split('\t')[1]
                line3=evaluation[2].strip('\n').split('\t')[1]
                line4=evaluation[3].strip('\n').split('\t')[1]
                #mrr+=float(line1)
                #median+=float(line2)
                list_var_mrr.append(float(line1))
                list_var_median.append(float(line2))
                list_var_mean.append(float(line3))
                plot_median[setup].append(float(line2))
                plot_mrr[setup].append(float(line1))
                #characters+=int(line3)
                characters.append(int(line4))
                #counter+=1
            if 'character' in single_file:
                characters_file=open('{}/{}/{}/{}'.format(big, setup, novel, single_file)).readlines()
                for l in characters_file:
                    l=l.split('\t') 
                    l=int(l[0])
                    if l>=10:
                        characters_frequency.append(l)
        original_file=os.listdir('{}/{}/{}/original_novel'.format(big, setup, novel))
        open_file=open('{}/{}/{}/original_novel/{}'.format(big, setup, novel, original_file[0])).read()
        open_file=sub(r'\W+', ' ', open_file)
        open_file=open_file.split(' ')
        novel_length=len(open_file)
        lengths[setup].append(novel_length)
        names[setup].append(novel)
        characters_dict[setup].append(int(line4))
        std_characters_frequency=numpy.std(characters_frequency)
        characters_std[setup].append(std_characters_frequency)
    #average_mrr=mrr/float(counter)
    average_mrr=numpy.median(list_var_mrr)
    var_mrr=numpy.var(list_var_mrr)
    #average_median=median/float(counter)
    average_median=numpy.median(list_var_median)
    var_median=numpy.var(list_var_median)
    average_mean=numpy.median(list_var_mean)
    var_mean=numpy.var(list_var_mean)
    #average_characters=characters/counter
    average_characters=numpy.mean(characters)
    print('Setup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}'.format(setup, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr)))

def ticks(setups_dict, mode):
    max_value=[]
    for i in setups_dict.keys():
        max_i=max(setups_dict[i])
        max_value.append(max_i) 
    x_ticks=[u for u in range(1, len(setups_dict[i])+1)]
    if mode=='median':
        y_ticks=[i for i in range(1, int(max(max_value))+1, 2)] 
    elif mode=='mrr':
        y_ticks=[i for i in numpy.linspace(0, 1, 11)] 
    return x_ticks, y_ticks

short_names=[]
for n in names[setup]:
    n=n.replace('_', ' ').split(' by')
    short_names.append(n[0])

### Novels data:

novels_info=open('plots/novels_info.txt', 'w')
for n in [k for k in range(0, len(short_names))]:
    novels_info.write('Name:\t{}\nLength in words:\t{}\nCharacters evaluated:\t{}\nMedian Rank:\t{}\nMRR:\t{}\nStandard deviation of characters frequency: {}\n\n'.format(short_names[n],lengths[setup][n],characters_dict[setup][n],list_var_median[n],list_var_mrr[n], characters_std[setup][n]))


### Lenghts/score

for setup in plot_median.keys():
    plt.scatter( plot_median[setup],lengths[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_median, 'median')
plt.xlabel('Median Rank')
plt.ylabel('Novel length (words)')
plt.yticks(lengths[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/lengths_median.png', dpi=900)

plt.clf()


### Number of characters/score

for setup in plot_median.keys():
    plt.scatter( plot_median[setup], characters_dict[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_median, 'median')
plt.xlabel('Median Rank')
plt.ylabel('Number of characters')
plt.yticks(characters_dict[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/characters_median.png', dpi=900)

plt.clf()


### Variance of characters frequency/score

for setup in plot_median.keys():
    plt.scatter( plot_median[setup], characters_std[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_median, 'median')
plt.xlabel('Median Rank')
plt.ylabel('Variance of characters frequency')
plt.yticks(characters_std[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/characters_frequency_median.png', dpi=1200)

plt.clf()

### Score/novel

for setup in plot_median.keys():
    plt.scatter(plot_median[setup], short_names, label='{}'.format(setup))
plt.xlabel('Median Rank')
plt.ylabel('Novel')
plt.xticks(y_ticks)
plt.yticks(short_names)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/plot_medians.png'.format(setup, novel), dpi=900)
plt.clf()

### Lenghts/score MRR

for setup in plot_mrr.keys():
    plt.scatter( plot_mrr[setup],lengths[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
plt.xlabel('MRR')
plt.ylabel('Novel length (words)')
plt.yticks(lengths[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/lengths_mrr.png', dpi=900)

plt.clf()


### Number of characters/score MRR

for setup in plot_mrr.keys():
    plt.scatter( plot_mrr[setup], characters_dict[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
plt.xlabel('MRR')
plt.ylabel('Number of characters')
plt.yticks(characters_dict[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/characters_MRR.png', dpi=900)

plt.clf()


### Variance of characters frequency/score MRR

for setup in plot_mrr.keys():
    plt.scatter( plot_mrr[setup], characters_std[setup], label='{}'.format(setup)) 
x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
plt.xlabel('MRR')
plt.ylabel('Variance of characters frequency')
plt.yticks(characters_std[setup])
plt.xticks(y_ticks)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/characters_frequency_mrr.png', dpi=1200)

plt.clf()

### Score mrr /novel

for setup in plot_mrr.keys():
    plt.scatter(plot_mrr[setup], short_names, label='{}'.format(setup))
plt.xlabel('MRR')
plt.ylabel('Novel')
plt.xticks(y_ticks)
plt.yticks(short_names)
plt.legend(bbox_to_anchor=(0.9, 1.15))
plt.tight_layout()
plt.savefig('plots/name_mrr.png'.format(setup, novel), dpi=900)
plt.clf()

