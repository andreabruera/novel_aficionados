import os
import numpy
import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import re
import scipy
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from re import sub

parser=argparse.ArgumentParser()
parser.add_argument('--training_mode')
args = parser.parse_args()

#from matplotlib import rcParams
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']

cwd=os.getcwd()
big='{}/{}_test_novels'.format(cwd, args.training_mode)
output_folder='{}_doppelgaenger_test_plots'.format(args.training_mode)
os.makedirs(output_folder, exist_ok=True)

plot_median={}
plot_mrr={}
lengths={}
names={}
characters_dict={}
characters_std={}
total_evaluations_runs_counter=0

for setup in os.listdir(big):
    
    if 'sum' in setup:
        setup_shorthand='sum'
    else:
        setup_shorthand='c2v'


    setup_folder=os.listdir('{}/{}'.format(big, setup))
    #median=[]
    #characters=0
    characters=[]
    list_var_mrr=[]
    list_var_median=[]
    list_var_mean=[]

    plot_median[setup]=[]
    plot_mrr[setup]=[]
    lengths[setup]=[]
    names[setup]=[]
    characters_dict[setup]=[]
    characters_std[setup]=[]

    ambiguities={}
    ambiguities_present=False
    
    for novel in setup_folder:
        sentences_counter=[]
        ambiguities_counter=[]
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
            if 'character' in single_file and 'ender' not in single_file:
                characters_file=open('{}/{}/{}/{}'.format(big, setup, novel, single_file)).readlines()
                for l in characters_file:
                    l=l.split('\t') 
                    l=int(l[0])
                    if l>=10:
                        characters_frequency.append(l)
            if 'data_output' in single_file:
                data_output_filenames=os.listdir('{}/{}/{}/data_output'.format(big, setup, novel))
                if 'ambiguities' in data_output_filenames:
                    ambiguities_present=True
                    ambiguities_filenames=os.listdir('{}/{}/{}/data_output/ambiguities'.format(big, setup, novel))
                    for ambiguity in ambiguities_filenames:
                        current_ambiguity=open('{}/{}/{}/data_output/ambiguities/{}'.format(big, setup, novel, ambiguity)).readlines()
                        for character_line in current_ambiguity:
                            if 'too: ' in character_line:
                                character_line=character_line.strip('\n').split('too: ')[1]
                                character_ambiguity=character_line.split(' out of ')[0]
                                sent=character_line.split(' out of ')[1].strip('\n').replace(' sentences', '')
                                ambiguities_counter.append(int(character_ambiguity))
                                sentences_counter.append(int(sent))
            if numpy.sum(sentences_counter)==0:
                ambiguities_present=False
        if ambiguities_present==True:
            novel_ambiguity=numpy.sum(ambiguities_counter)
            total_sentences=numpy.sum(sentences_counter)
            percentage=round((float(novel_ambiguity)*100.0)/float(total_sentences), 2)
            ambiguities[novel]=[novel_ambiguity, total_sentences, percentage]   

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
    average_mrr=numpy.median(list_var_mrr)
    var_mrr=numpy.var(list_var_mrr)
    average_median=numpy.median(list_var_median)
    var_median=numpy.var(list_var_median)
    average_mean=numpy.median(list_var_mean)
    var_mean=numpy.var(list_var_mean)
    average_characters=numpy.mean(characters)
    if average_characters>14.0:
        print('Setup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}'.format(setup, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr)))
        results_output=open('{}/doppel_results_{}.txt'.format(output_folder, setup),'w')
        results_output.write('Setup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}'.format(setup, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr)))
        if ambiguities_present==True:
            if len(ambiguities.keys())>0 and total_evaluations_runs_counter==0:
                ambiguity_percentages=[]
                for amb in ambiguities.keys():
                    novel_amb=ambiguities[amb]
                    amb_sent=novel_amb[0]
                    total_sent=novel_amb[1]
                    percent=novel_amb[2]
                    ambiguity_percentages.append(percent)
                final_percent=numpy.mean(ambiguity_percentages)
                print('Percentage of ambiguous sentences of all sentences used for training (containing more than one character): {} %\n'.format(round(final_percent, 3)))
                total_evaluations_runs_counter+=1



    ### Ambiguity infos


    with open('{}/doppel_ambiguities_info.txt'.format(output_folder), 'w') as ambiguity_file:
        if len(ambiguities.keys())>0:
            for amb in ambiguities.keys():
                novel_amb=ambiguities[amb]
                amb_sent=novel_amb[0]
                total_sent=novel_amb[1]
                perc_amb=novel_amb[2]
                ambiguity_file.write('\nAmbiguous sentences: {} out of {}\nPercentage: {} %\n\n'.format(amb_sent, total_sent, perc_amb))

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

    sum_spearman='N.A.'

    ### Novels data:

    novels_info=open('{}/{}_novels_info.txt'.format(output_folder, setup_shorthand), 'w')
    for n in [k for k in range(0, len(short_names))]:
        novels_info.write('Name:\t{}\nLength in words:\t{}\nCharacters evaluated:\t{}\nMedian Rank:\t{}\nMRR:\t{}\nStandard deviation of characters frequency: {}\n\n'.format(short_names[n],lengths[setup][n],characters_dict[setup][n],list_var_median[n],list_var_mrr[n], characters_std[setup][n]))

    sum_color=['lightslategrey', 'coral', 'darkgoldenrod', 'darkcyan', 'deeppink', 'lightgreen', 'aquamarine', 'green', 'purple', 'gold', 'sienna', 'olivedrab']

    ### Lenghts/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup],lengths[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], lengths[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup],lengths[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2V: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], lengths[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Novel length (words)')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.yticks([(((i+999)/1000)*1000) for i in numpy.linspace(0,max(lengths[setup]),10)])
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_lenghts.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Number of characters/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], characters_dict[setup], color=sum_color, label=legend_label, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_dict[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], characters_dict[setup], color=sum_color, label=legend_label, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_dict[setup])[0],2)) 
    x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Number of characters')
    #plt.yticks(characters_dict[setup] )
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_characters.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Variance of characters frequency/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], characters_std[setup], label=legend_label, color=sum_color, marker= 'P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_std[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], characters_std[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_std[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Variance of character mention frequency')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.yticks(characters_std[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_frequency_std.eps'.format(output_folder), dpi=1200, format='eps', pad_inches=0.2, bbox_inches='tight')

    plt.clf()

    ### Score/novel

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], short_names, label=legend_label, color=sum_color, marker='P')
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], short_names, label=legend_label, color=sum_color, marker='v')
    plt.xlabel('Median Rank')
    plt.ylabel('Novel')
    #plt.xticks(y_ticks)
    #plt.yticks(short_names )
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_names.eps'.format(output_folder, setup, novel), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)
    plt.clf()

    ### Lenghts/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup],lengths[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], lengths[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup],lengths[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], lengths[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Novel length (words)')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(lengths[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_mrr_lengths.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Number of characters/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup], characters_dict[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_dict[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup], characters_dict[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_dict[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Number of characters')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(characters_dict[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_characters.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Variance of characters frequency/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup], characters_std[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], characters_std[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup], characters_std[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_std[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Variance of character mention frequency')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(characters_std[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_characters_std.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)

    plt.clf()

    ### Score mrr /novel

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_mrr[setup], short_names, label=legend_label, color=sum_color, marker='P')
        else:
            legend_label='N2V'
            plt.scatter(plot_mrr[setup], short_names, label=legend_label, color=sum_color, marker='v')
    plt.xlabel('MRR')
    plt.ylabel('Novel')
    plt.gca().invert_xaxis()
    #plt.xticks(y_ticks)
    #plt.yticks(short_names )
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_names.eps'.format(output_folder), dpi=1200, format='eps', bbox_inches='tight', pad_inches=0.2)
    plt.clf()
