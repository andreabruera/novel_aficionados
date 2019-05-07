"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os
### NOVELS EDIT: added pickle, and then the sys & io modules to capture the sysout, thus capturing the alpha
import sys, io, pickle

import argparse
import logging
import logging.config

import math
import scipy
import numpy as np
import collections
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import gensim

from gensim.models import Word2Vec

import nonce2vec.utils.novels_utilities

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils

from nonce2vec.models.nonce2vec import Nonce2Vec, Nonce2VecVocab, \
                                       Nonce2VecTrainables
### NOVELS EDIT: importing the models needed for the novels training
from nonce2vec.models.nonce2vec import Nonce2Vec_novels, Nonce2VecVocab_novels, \
                                       Nonce2VecTrainables_novels

from nonce2vec.models.BERT_for_novels import BERT_test

from nonce2vec.utils.files import Samples

from nonce2vec.utils.get_damn_evaluation import NovelsEvaluation

from collections import defaultdict

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def _get_rank(probe, nns):
    for idx, nonce_similar_word in enumerate(nns):
        word = nonce_similar_word[0]
        if word == probe:
            rank = idx + 1  # rank starts at 1
    if not rank:
        raise Exception('Could not find probe {} in nonce most similar words '
                        '{}'.format(probe, nns))
    return rank


def _update_rr_and_count(relative_ranks, count, rank):
    relative_rank = 1.0 / float(rank)
    relative_ranks += relative_rank
    count += 1
    logger.info('Rank, Relative Rank = {} {}'.format(rank, relative_rank))
    logger.info('MRR = {}'.format(relative_ranks/count))
    return relative_ranks, count


def _load_nonce2vec_model(args, nonce):
    logger.info('Loading Nonce2Vec model...')
    ### NOVELS EDIT: added this condition, which loads the original modules for n2v if you want to test it NOT on novels
    if 'novel' not in args.on:
        model = Nonce2Vec.load(args.background)
        model.vocabulary = Nonce2VecVocab.load(model.vocabulary)
        model.trainables = Nonce2VecTrainables.load(model.trainables)
    ### NOVELS EDIT: added this condition, which calls the novels version of the functions and classes
    elif args.on == 'novels' or args.on == 'wiki_novel' or args.on == 'wiki_novels':
        logger.info('Testing on:     novels')
        model = Nonce2Vec_novels.load(args.background)
        model.vocabulary = Nonce2VecVocab_novels.load(model.vocabulary)
        model.trainables = Nonce2VecTrainables_novels.load(model.trainables)
    model.sg = 1
    model.min_count = 5  # min_count should be the same as the background model!!
    model.sample = args.sample
    logger.info('Running original n2v code for replication...')
    if args.sample is None:
        raise Exception('In replication mode you need to specify the '
                        'sample parameter')
    if args.window_decay is None:
        raise Exception('In replication mode you need to specify the '
                        'window_decay parameter')
    if args.sample_decay is None:
        raise Exception('In replication mode you need to specify the '
                        'sample_decay parameter')
    ### NOVELS EDIT: added the next line because the parameters didn't seem to be passed on
    model.sample_decay = args.sample_decay
    ### NOVELS EDIT: added the next line because the parameters didn't seem to be passed on
    model.window_decay = args.window_decay
    ### NOVELS EDIT: added the next line because the parameters didn't seem to be passed on
    model.window = args.window
    if not args.sum_only:
        model.train_with = args.train_with
        model.alpha = args.alpha
        model.iter = args.epochs
        model.negative = args.neg
        model.lambda_den = args.lambda_den
        model.neg_labels = []
        if model.negative > 0:
            # precompute negative labels optimization for pure-python training
            model.neg_labels = np.zeros(model.negative + 1)
            model.neg_labels[0] = 1.
    #model.trainables.info = info
    model.workers = args.num_threads
    model.vocabulary.nonce = nonce
    ### NOVELS EDIT: added the sentence_count counter, which resets to 0 every time you load a model (i.e. every time you start a training session on novels)
    if args.on=='novels' or args.on=='wiki_novel' or args.on=='wiki_novels':
        model.sentence_count=0
    logger.info('Model loaded with lambda_den= {}, window_decay= {} and sample_decay= {}'. format(model.lambda_den, model.sample_decay, model.window_decay))
    return model

def _test_on_chimeras(args):
    nonce = '[MASK]'
    rhos = []
    count = 0
    samples = Samples(args.dataset, source='chimeras')
    total_num_batches = sum(1 for x in samples)
    total_num_sent = sum(1 for x in [sent for batch in samples for sent in batch])
    logger.info('Testing Nonce2Vec on the chimeras dataset containing '
                '{} batches and {} sentences'.format(total_num_batches,
                                                     total_num_sent))
    num_batch = 1
    for sentences, probes, responses in samples:
        logger.info('-' * 30)
        logger.info('Processing batch {}/{}'.format(num_batch,
                                                    total_num_batches))
        num_batch += 1
        logger.info('sentences = {}'.format(sentences))
        logger.info('probes = {}'.format(probes))
        logger.info('responses = {}'.format(responses))
        model = _load_nonce2vec_model(args, nonce)
        model.vocabulary.nonce = '[MASK]'
        # A quick and dirty bugfix to add the nonce to the vocab
        # model.wv.vocab['[MASK]'] = Vocab(count=1,
        #                               index=len(model.wv.index2word))
        # model.wv.index2word.append('[MASK]')
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
        model.build_vocab(sentences, update=True)
        if not args.sum_only:
            model.train(sentences, total_examples=model.corpus_count,
                        epochs=model.iter)
        system_responses = []
        human_responses = []
        probe_count = 0
        for probe in probes:
            try:
                cos = model.similarity(nonce, probe)
                system_responses.append(cos)
                human_responses.append(responses[probe_count])
            except:
                logger.error('ERROR processing probe {}'.format(probe))
            probe_count += 1
        if len(system_responses) > 1:
            logger.info('system_responses = {}'.format(system_responses))
            logger.info('human_responses = {}'.format(human_responses))
            logger.info('10 most similar words = {}'.format(
                model.most_similar(nonce, topn=10)))
            rho = _spearman(human_responses, system_responses)
            logger.info('RHO = {}'.format(rho))
            if not math.isnan(rho):
                rhos.append(rho)
        count += 1
    logger.info('AVERAGE RHO = {}'.format(float(sum(rhos))/float(len(rhos))))


def _compute_average_sim(sims):
    sim_sum = sum(sim[1] for sim in sims)
    return sim_sum / len(sims)


def _test_on_nonces(args):
    """Test the definitional nonces with a one-off learning procedure."""
    ranks = []
    sum_10 = []
    sum_25 = []
    sum_50 = []
    relative_ranks = 0.0
    count = 0
    samples = Samples(args.dataset, source='nonces')
    total_num_sent = sum(1 for line in samples)
    logger.info('Testing Nonce2Vec on the nonces dataset containing '
                '{} sentences'.format(total_num_sent))
    num_sent = 1
    for sentences, nonce, probe in samples:
        logger.info('-' * 30)
        logger.info('Processing sentence {}/{}'.format(num_sent,
                                                       total_num_sent))
        model = _load_nonce2vec_model(args, nonce)
        model.vocabulary.nonce = nonce
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
        logger.info('nonce: {}'.format(nonce))
        logger.info('sentence: {}'.format(sentences))
        if nonce not in model.wv.vocab:
            logger.error('Nonce \'{}\' not in gensim.word2vec.model '
                         'vocabulary'.format(nonce))
            continue
        model.build_vocab(sentences, update=True)
        if not args.sum_only:
            model.train(sentences, total_examples=model.corpus_count,
                        epochs=model.iter)
        nns = model.most_similar(nonce, topn=vocab_size)
        logger.info('10 most similar words: {}'.format(nns[:10]))
        rank = _get_rank(probe, nns)
        relative_ranks, count = _update_rr_and_count(relative_ranks, count,
                                                     rank)
        num_sent += 1
    logger.info('Final MRR =  {}'.format(relative_ranks/count))

def _get_men_pairs_and_sim(men_dataset):
    pairs = []
    humans = []
    with open(men_dataset, 'r') as men_stream:
        for line in men_stream:
            line = line.rstrip('\n')
            items = line.split()
            pairs.append((items[0], items[1]))
            humans.append(float(items[2]))
    return pairs, humans


def _cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


def _check_men(args):
    """Check embeddings quality.

    Calculate correlation with the similarity ratings in the MEN dataset.
    """
    logger.info('Checking embeddings quality against MEN similarity ratings')
    pairs, humans = _get_men_pairs_and_sim(args.men_dataset)
    logger.info('Loading word2vec model...')
    model = Word2Vec.load(args.w2v_model)
    logger.info('Model loaded')
    system_actual = []
    human_actual = []  # This is needed because we may not be able to
                       # calculate cosine for all pairs
    count = 0
    for (first, second), human in zip(pairs, humans):
        if first not in model.wv.vocab or second not in model.wv.vocab:
            logger.error('Could not find one of more pair item in model '
                         'vocabulary: {}, {}'.format(first, second))
            continue
        sim = _cosine_similarity(model.wv[first], model.wv[second])
        system_actual.append(sim)
        human_actual.append(human)
        count += 1
    spr = _spearman(human_actual, system_actual)
    logger.info('SPEARMAN: {} calculated over {} items'.format(spr, count))


def _train(args):
    sentences = Samples(args.datadir, source='wiki')
    if not args.train_mode:
        raise Exception('Unspecified train mode')
    output_model_filepath = futils.get_model_path(args.datadir, args.outputdir,
                                                  args.train_mode,
                                                  args.alpha, args.neg,
                                                  args.window, args.sample,
                                                  args.epochs,
                                                  args.min_count, args.size)
    model = gensim.models.Word2Vec(
        min_count=args.min_count, alpha=args.alpha, negative=args.neg,
        window=args.window, sample=args.sample, iter=args.epochs,
        size=args.size, workers=args.num_threads)
    if args.train_mode == 'cbow':
        model.sg = 0
    if args.train_mode == 'skipgram':
        model.sg = 1
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(output_model_filepath)


def _test(args):
    if args.on == 'chimeras':
        _test_on_chimeras(args)
    elif args.on == 'nonces':
        _test_on_nonces(args)
    elif args.on == 'novels' or args.on == 'novel':
        test_on_novel(args)
    elif args.on == 'wiki_novels' or args.on == 'wiki_novel':
        test_on_wiki_novel(args)
    elif args.on == 'bert_novels':
        test_on_bert_novel(args)


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    # a shared set of parameters when using gensim
    parser_gensim = argparse.ArgumentParser(add_help=False)
    parser_gensim.add_argument('--num-threads', type=int, default=1,
                               help='number of threads to be used by gensim')
    parser_gensim.add_argument('--alpha', type=float,
                               help='initial learning rate')
    parser_gensim.add_argument('--neg', type=int,
                               help='number of negative samples')
    parser_gensim.add_argument('--window', type=int,
                               help='window size')
    parser_gensim.add_argument('--sample', type=float,
                               help='subsampling rate')
    parser_gensim.add_argument('--epochs', type=int,
                               help='number of epochs')
    parser_gensim.add_argument('--min-count', type=int,
                               help='min frequency count')

    # train word2vec with gensim from a wikipedia dump
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim],
        help='generate pre-trained embeddings from wikipedia dump via '
             'gensim.word2vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True, dest='datadir',
                              help='absolute path to training data directory')
    parser_train.add_argument('--size', type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--train-mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='Absolute path to outputdir to save model')

    # check various metrics
    parser_check = subparsers.add_parser(
        'check', formatter_class=argparse.RawTextHelpFormatter,
        help='check w2v embeddings quality by calculating correlation with '
             'the similarity ratings in the MEN dataset. Also, check the '
             'distribution of context_entropy across datasets')
    parser_check.set_defaults(func=_check_men)
    parser_check.add_argument('--data', required=True, dest='men_dataset',
                              help='absolute path to dataset')
    parser_check.add_argument('--model', required=True, dest='w2v_model',
                              help='absolute path to the word2vec model')

    # test nonce2vec in various config on the chimeras and nonces datasets
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim],
        help='test nonce2vec')
    parser_test.set_defaults(func=_test)
    ### NOVELS EDIT: added the dest for this argument, because it's needed for calling the novels model of N2V or the basic one
    parser_test.add_argument('--on', required=True,
                             choices=['nonces', 'chimeras','novels','novel','wiki_novel','wiki_novels', 'bert_novels'],
                             help='type of test data to be used')
    parser_test.add_argument('--model', required=True,
                             dest='background',
                             help='absolute path to word2vec pretrained model')
    parser_test.add_argument('--data', required=True, dest='dataset',
                             help='absolute path to test dataset')
    ### NOVELS EDIT: added this argument, which is needed for getting the right folder
    parser_test.add_argument('--folder', required=False,
                             dest='folder',
                             help='absolute path to the novel folder')
    parser_test.add_argument('--simil_out', required=False, default='False',
                             action='store_true',
                             help='specify whether it is needed to write a file with top similarities')
    parser_test.add_argument('--train-with', required = False, default = 1,
                             choices=['exp_alpha'],
                             help='learning rate computation function')
    parser_test.add_argument('--lambda', type=float, required = False, default = 70.0,
                             dest='lambda_den',
                             help='lambda decay')
    ### NOVELS_EDIT: added the destination with the '_' because it didn't seem to be present in the original code - and this messes everything up!
    parser_test.add_argument('--sample-decay', type=float, required = False,
                             default = 1.1, dest='sample_decay',
                             help='sample decay')
    ### NOVELS_EDIT: added the destination with the '_' because it didn't seem to be present in the original code - and this messes everything up!
    parser_test.add_argument('--window-decay', type=int, required = False, default = 3, 
                             dest='window_decay',
                             help='window decay')
    ### NOVELS_EDIT: changed 'sum-only' to 'sum_only', added 'required=False'
    parser_test.add_argument('--sum_only', required=False, action='store_true', default=False,
                             help='sum only: no additional training after '
                                  'sum initialization')
    ### NOVELS_EDIT: added this for training on the wikipedia pages
    parser_test.add_argument('--quality_test', type = bool, required = False, default = False)
    ### NOVELS_EDIT: added this optional argument for testing the random selection of the sentences
    parser_test.add_argument('--random_sentences', required=False, type = bool, default=False, help='random_sentences: instead of picking sentences the original order it picks up sentences randomly')
    parser_test.add_argument('--bert_layers', required = False, type = int, default = 2)
    args = parser.parse_args()
    args.func(args)

#########################################################################################

def test_on_novel(args):
    #char_list=get_characters_list(args.folder, args.dataset)
    books_dict=get_books_dict(args.dataset)
    
    char_dict={} 

    nonce='[MASK]'
    
    folder=args.folder
    temp_novel_folder='{}/temp'.format(folder)
    data_output_folder='{}/data_output'.format(folder)

    os.makedirs('{}'.format(data_output_folder),exist_ok=True)
    if args.simil_out==True:
        os.makedirs('{}/details'.format(data_output_folder), exist_ok=True)
        os.makedirs('{}/ambiguities'.format(data_output_folder), exist_ok=True)

    clean_novel_name='{}/{}_clean.txt'.format(temp_novel_folder, args.dataset)

    novel_versions, current_char_list, background_vocab, genders_dict = prepare_for_n2v(args.folder, args.dataset, clean_novel_name, args.background, write_to_file=True)

    novel_versions_keys=novel_versions.keys()

    #for part in books_dict.keys():
    for path in novel_versions_keys:
        if 'part_a' in path:
            part='a'
        elif 'part_b' in path:
            part='b'
        else:
            part='full'
    
        #filename=books_dict[part]
        version=novel_versions[path]

        for character in current_char_list:

            character_counter={}
            
            for key in novel_versions_keys:
                character_counter[key]=0
                for line in novel_versions[key]:
                    if character in line:
                        character_counter[key]+=1

            if 0 not in [character_counter[i] for i in character_counter.keys()]:

                if genders_dict[character]=='FEMALE':
                    prototype='woman'
                elif genders_dict[character]=='MALE':
                    prototype='man'
                else:
                    prototype=nonce 
                model=_load_nonce2vec_model(args, prototype)
                model.vocabulary.nonce=nonce
                model.sentence_count=0


                sent_list, sent_vocab_list = get_novel_sentences_from_versions_dict(version, character, background_vocab)

                logger.info('List of sentences created!')
                logger.info('Number of total sentences: {}'.format(len(sent_list)))

                if args.simil_out==True and len(sent_list)>10 and len(sent_list)<10000:

                    simil_out_file=open('{}/details/{}_{}.similarities'.format(data_output_folder, character, part),'w')
                    simil_out_file.write('{} part {}\n\n'.format(character, part))
                    ambiguity_out_file=open('{}/ambiguities/{}_ambiguity_detector_part_{}'.format(data_output_folder, character, part), 'w')
                    ambiguity_out_file.write('{} part {}\n\n'.format(character, part))

                ambiguity_detector={}

                for alias in current_char_list:
                    ambiguity_detector[alias]=0
 
                if args.random_sentences==True:
                    indexes=np.random.choice(len(sent_list), len(sent_list), replace=False)
                else:
                    indexes=np.arange(len(sent_list))

                for index in indexes:

                    sentence=sent_list[index]

                    if '[MASK]' in sentence and len(sentence)>3:

                        model.sentence_count+=1
                        for alias in current_char_list:
                            if alias in sentence and alias != character:
                                ambiguity_detector[alias]+=1
                                sentence.remove(alias)
                        ### NOVELS_EDIT: added the <50 condition in order to reduce training time and to give more balanced training to every character.
                        if not args.sum_only and model.sentence_count<=50:
                            logger.info('Current subsampling rate: {}'.format(model.sample))
                            vocab_sentence=[sentence]
                            logger.info('Full sentence: {}'.format(vocab_sentence))
                            model.build_vocab(vocab_sentence, model.sentence_count, update=True)
                            vocab_size=len(model.wv.vocab)
                            logger.info('Current nonce: {}'.format(character))

                            ### NOVELS NOTE: this is the part where the training happens
                            model.train(vocab_sentence, total_examples=model.corpus_count,epochs=model.iter)

                            top_similarities=model.most_similar(nonce,topn=20)
                            logger.info('Top similarities for {} at this point during the training: {}'.format(character, top_similarities))

                            simil_out_file.write('Sentence no. {}\nSentence: {}\n{}\n\n'.format(model.sentence_count, sentence, model.wv.most_similar(nonce, topn=50))) 
                            if model.sample > 10:
                                model.sample = model.sample / args.sample_decay
                        elif model.sentence_count>49:
                            break
                        elif args.sum_only:
                            if model.sentence_count==1:
                                list_of_sentences=[]
                            if model.sentence_count<50:
                                list_of_sentences.append(sentence)
                                summed_sentences=model.sentence_count
                            elif model.sentence_count>49:
                                break

                if args.sum_only and model.sentence_count>0:
                    list_of_words=[]
                    stopwords=["","(",")","a","about","an","and","are","around","as","at","away","be","become","became","been","being","by","did","do","does","during","each","for","from","get","have","has","had","her","his","how","i","if","in","is","it","its","made","make","many","most","of","on","or","s","some","that","the","their","there","this","these","those","to","under","was","were","what","when","where","who","will","with","you","your"]
                    for s in list_of_sentences:
                        for w in s:
                            if w not in stopwords:
                                list_of_words.append(w)

                    model.sentence_count=1
                    sentence=list_of_words
                    logger.info('Current subsampling rate: {}'.format(model.sample))
                    vocab_sentence=[sentence]
                    logger.info('Full sentence: {}'.format(vocab_sentence))
                    model.build_vocab(vocab_sentence, model.sentence_count, update=True)
                    vocab_size=len(model.wv.vocab)
                    logger.info('Current nonce: {}'.format(character))
                    ### NOVELS NOTE: this is the part where the training happens
                    top_similarities=model.most_similar(nonce,topn=20)
                    logger.info('Top similarities for {} at this point during the training: {}'.format(character, top_similarities))
                    model.sentence_count=summed_sentences

                    simil_out_file.write('Sentence no. {}\nSentence: {}\n{}\n\n'.format(model.sentence_count, sentence, model.wv.most_similar(nonce, topn=50))) 
                    ### NOVELS NOTE: added the condition >0, because in the absence of a character in a certain part of the book, the same vector is created for all the absent characters and this creates fake 1.0 similarities at evaluation time.
                if model.sentence_count>0:
                    character_vector=model[nonce]
                    character_name_and_part='{}_{}'.format(character, part)
                    char_dict[character_name_and_part]=character_vector
                else:
                    os.remove('{}/details/{}_{}.similarities'.format(data_output_folder, character, part))
                if args.simil_out==True and model.sentence_count>0:
                    for alias in ambiguity_detector.keys():
                        
                        ambiguity_out_file.write('Character: {}\tNumber of sentences containing this character too: {} out of {} sentences\n\n'.format(alias, ambiguity_detector[alias], model.sentence_count))


    folder=args.folder
    logger.info('Length of the characters list: {}\n Characters list: {}\n'.format(len(char_dict.keys()), char_dict.keys()))
    with open('{}/{}.pickle'.format(data_output_folder, args.dataset),'wb') as out:        
        pickle.dump(char_dict,out,pickle.HIGHEST_PROTOCOL) 


#########################################################################################

def test_on_wiki_novel(args):

    char_dict={} 

    nonce='[MASK]'
    
    folder=args.folder
    data_output_folder='{}/quality_test/data_output'.format(folder)

    os.makedirs('{}'.format(data_output_folder),exist_ok=True)

    if args.simil_out==True:
        os.makedirs('{}/details'.format(data_output_folder), exist_ok=True)
        os.makedirs('{}/ambiguities'.format(data_output_folder), exist_ok=True)

    clean_novel_name='{}/quality_test/original_text/{}.txt'.format(folder, args.dataset)

    novel_versions, current_char_list, genders_dict = prepare_for_n2v(args.folder, args.dataset, clean_novel_name, write_to_file=True, wiki_novel=True)

    
    version=novel_versions

    for character in current_char_list:

        model=_load_nonce2vec_model(args, nonce)
        model.vocabulary.nonce=nonce
        model.sentence_count=0

        sent_list, sent_vocab_list = get_novel_sentences_from_versions_dict(version, character)
        logger.info('List of sentences created!')
        logger.info('Number of total sentences: {}'.format(len(sent_list)))

        if args.simil_out==True and len(sent_list)>10 and len(sent_list)<10000:

            simil_out_file=open('{}/details/{}.similarities'.format(data_output_folder, character),'w')
            simil_out_file.write('{}\n\n'.format(character))
            ambiguity_out_file=open('{}/ambiguities/{}_ambiguity_detector'.format(data_output_folder, character), 'w')
            ambiguity_out_file.write('{}\n\n'.format(character))

        ambiguity_detector={}

        for alias in current_char_list:
            ambiguity_detector[alias]=0

        for index,sentence in enumerate(sent_list):

            if '[MASK]' in sentence and len(sentence)>=3:

                model.sentence_count+=1
                for alias in current_char_list:
                    if alias in sentence and alias != character:
                        ambiguity_detector[alias]+=1
                        sentence.remove(alias)
                ### NOVELS_EDIT: added the <50 condition in order to reduce training time and to give more or less equal training to every character.
                if not args.sum_only and model.sentence_count<=50:
                    logger.info('Current subsampling rate: {}'.format(model.sample))
                    vocab_sentence=[sentence]
                    logger.info('Full sentence: {}'.format(vocab_sentence))
                    model.build_vocab(vocab_sentence, model.sentence_count, update=True)
                    vocab_size=len(model.wv.vocab)
                    logger.info('Current nonce: {}'.format(character))
                    ### NOVELS NOTE: this is the part where the training happens
                    if not args.sum_only:
                        model.train(vocab_sentence, total_examples=model.corpus_count,epochs=model.iter)
                    top_similarities=model.most_similar(nonce,topn=20)
                    logger.info('Top similarities for {} at this point during the training: {}'.format(character, top_similarities))

                    simil_out_file.write('Sentence no. {}\nSentence: {}\n{}\n\n'.format(model.sentence_count, sentence, model.wv.most_similar(nonce, topn=50))) 
                    if model.sample > 10:
                        model.sample = model.sample / args.sample_decay
                elif model.sentence_count>49:
                    break
                elif args.sum_only:
                    if model.sentence_count==1:
                        list_of_sentences=[]
                    if model.sentence_count<50:
                        list_of_sentences.append(sentence)
                        summed_sentences=model.sentence_count
                    elif model.sentence_count>49:
                        break

        if args.sum_only and model.sentence_count>0:
            list_of_words=[]
            stopwords=["","(",")","a","about","an","and","are","around","as","at","away","be","become","became","been","being","by","did","do","does","during","each","for","from","get","have","has","had","her","his","how","i","if","in","is","it","its","made","make","many","most","of","on","or","s","some","that","the","their","there","this","these","those","to","under","was","were","what","when","where","who","will","with","you","your"]
            for s in list_of_sentences:
                for w in s:
                    if w not in stopwords:
                        list_of_words.append(w)

            model.sentence_count=1
            sentence=list_of_words
            logger.info('Current subsampling rate: {}'.format(model.sample))
            vocab_sentence=[sentence]
            logger.info('Full sentence: {}'.format(vocab_sentence))
            model.build_vocab(vocab_sentence, model.sentence_count, update=True)
            vocab_size=len(model.wv.vocab)
            logger.info('Current nonce: {}'.format(character))
            ### NOVELS NOTE: this is the part where the training happens
            top_similarities=model.most_similar(nonce,topn=20)
            logger.info('Top similarities for {} at this point during the training: {}'.format(character, top_similarities))

            model.sentence_count=summed_sentences

            simil_out_file.write('Sentence no. {}\nSentence: {}\n{}\n\n'.format(model.sentence_count, sentence, model.wv.most_similar(nonce, topn=50))) 
            ### NOVELS NOTE: added the condition >0, because in the absence of a character in a certain part of the book, the same vector is created for all the absent characters and this creates fake 1.0 similarities at evaluation time.
        if model.sentence_count>0:
            character_vector=model[nonce]
            character_name_and_part='{}'.format(character)
            char_dict[character_name_and_part]=character_vector
        else:
            os.remove('{}/details/{}.similarities'.format(data_output_folder, character))
        if args.simil_out==True and model.sentence_count>0:
            for alias in ambiguity_detector.keys():
                
                ambiguity_out_file.write('Character: {}\tNumber of sentences containing this character too: {} out of {} sentences\n\n'.format(alias, ambiguity_detector[alias], model.sentence_count))


    folder=args.folder
    logger.info('Length of the characters list: {}\n Characters list: {}\n'.format(len(char_dict.keys()), char_dict.keys()))
    with open('{}/wiki_{}.pickle'.format(data_output_folder, args.dataset),'wb') as out:        
        pickle.dump(char_dict,out,pickle.HIGHEST_PROTOCOL) 

###############################################################################


def test_on_bert_novel(args):

    logger.info('Training on BERT')

    books_dict=get_books_dict(args.dataset)
    
    char_dict = defaultdict(list) 

    nonce='[MASK]'
    
    folder=args.folder
    temp_novel_folder='{}/temp'.format(folder)
    data_output_folder='{}/data_output'.format(folder)

    os.makedirs('{}'.format(data_output_folder),exist_ok=True)

    clean_novel_name='{}/{}_clean.txt'.format(temp_novel_folder, args.dataset)

    novel_versions, current_char_list, genders_dict = prepare_for_bert(args.folder, args.dataset, clean_novel_name, write_to_file=True)

    novel_versions_keys=novel_versions.keys()

    logger.info('Length of the characters list: {}\n Characters list: {}\n'.format(len(current_char_list), current_char_list))

    for path in novel_versions_keys:
        if 'part_a' in path:
            part='a'
        elif 'part_b' in path:
            part='b'
        else:
            part='full'
    
        version=novel_versions[path]

        for character in current_char_list:

            character_counter={}
            
            for key in novel_versions_keys:
                character_counter[key]=0
                for line in novel_versions[key]:
                    if character in line:
                        character_counter[key]+=1

            character_name_and_part='{}_{}'.format(character, part)
            logger.info('Current nonce: {} - for part: {}'.format(character, part))

            if 0 not in [character_counter[i] for i in character_counter.keys()]:

                if genders_dict[character]=='FEMALE':
                    prototype='woman'
                elif genders_dict[character]=='MALE':
                    prototype='man'
                else:
                    prototype=nonce 

                sentence_count = 0

                sent_list, sent_vocab_list = get_novel_sentences_from_versions_dict_bert(version, character)

                logger.info('Number of sentences in this part of the book: {}'.format(len(sent_list)))

                if args.random_sentences == True:
                    indexes=np.random.choice(len(sent_list), len(sent_list), replace=False)
                else:
                    indexes=np.arange(len(sent_list))

                model = BertModel.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                for index in indexes:

                    sentence=sent_list[index]

                    if '[MASK]' in sentence and len(sentence)>3:

                        sentence_count+=1

                        ### NOVELS_EDIT: added the <50 condition in order to reduce training time and to give more balanced training to every character.
                        if not args.sum_only and sentence_count<=50:

                            ### NOVELS NOTE: this is the part where the training happens
                            if sentence_count == 1:
                                char_dict[character_name_and_part] = BERT_test.train(args, sentence, character, model, tokenizer)    
                            elif sentence_count > 1:
                                char_dict[character_name_and_part] = BERT_test.train(args, sentence, character, model, tokenizer)    
                #if sentence_count >= 1:
                    #char_dict[character_name_and_part] = char_dict[character_name_and_part].numpy()

    folder=args.folder
    with open('{}/{}.pickle'.format(data_output_folder, args.dataset),'wb') as out:        
        pickle.dump(char_dict,out,pickle.HIGHEST_PROTOCOL) 
    evaluation = NovelsEvaluation()
    evaluation.bert_evaluation(folder)
