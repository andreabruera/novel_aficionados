### NOVELS EDIT: This is the edited version of Nonce2Vec, with the classes for testing the novels stored separately and marked with a '_novels' mark at the end of the class name. The original versions of the functions and of the classes are stored in the first part of this script. The '_novels' versions of the classes and functions are saved in the second part of this script.
"""
Nonce2Vec model.
A modified version of gensim.Word2Vec.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.special import expit
from six import iteritems
from six.moves import xrange
from gensim.models.word2vec import Word2Vec, Word2VecVocab, Word2VecTrainables
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import Vocab

__all__ = ('Nonce2Vec')

logger = logging.getLogger(__name__)

#####################################################################################

### START OF THE ORIGINAL FUNCTIONS AND CLASSES OF N2V

def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * np.exp(exp_decay) > min_alpha:
        return alpha * np.exp(exp_decay)
    return min_alpha

def train_sg_pair(model, word, context_index, alpha, nonce_count,
                  learn_vectors=True, learn_hidden=True, context_vectors=None,
                  context_locks=None, compute_loss=False, is_ft=False):
    if context_vectors is None:
        #context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        #context_locks = model.syn0_lockf
        context_locks = model.trainables.vectors_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]
        lambda_den = model.lambda_den
        exp_decay = -(nonce_count-1) / lambda_den
        if alpha * np.exp(exp_decay) > model.min_alpha:
            alpha = alpha * np.exp(exp_decay)
        else:
            alpha = model.min_alpha
        logger.info('training on \'{}\' and \'{}\' with '
                     'alpha = {}'.format(model.vocabulary.nonce,
                                         word,
                                         round(alpha, 5)))
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = np.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += np.outer(gb, l1)
                # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    result = 0
    window = model.window
    # Count the number of times that we see the nonce
    nonce_count = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in
                       model.wv.vocab and model.wv.vocab[w].sample_int
                       > model.random.rand() * 2 ** 32 or w == '[MASK]']
        for pos, word in enumerate(word_vocabs):
            # Note: we have got rid of the random window size
            start = max(0, pos - window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + window + 1)],
                                         start):
                # don't train on the `word` itself
                if pos2 != pos:
                    # If training context nonce, increment its count
                    if model.wv.index2word[word2.index] == \
                     model.vocabulary.nonce:
                        nonce_count += 1
                        train_sg_pair(
                            model, model.wv.index2word[word.index],
                            word2.index, alpha, nonce_count,
                            compute_loss=compute_loss)

        result += len(word_vocabs)
        if window - 1 >= 3:
            window = window - model.window_decay
        model.recompute_sample_ints()
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
        super(Nonce2VecVocab, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    @classmethod
    def load(cls, w2v_vocab):
        """Load a Nonce2VecVocab instance from a Word2VecVocab instance."""
        n2v_vocab = cls()
        for key, value in w2v_vocab.__dict__.items():
            setattr(n2v_vocab, key, value)
        return n2v_vocab

    def prepare_vocab(self, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            raise Exception('Nonce2Vec can only update a pre-existing '
                            'vocabulary')
        logger.info('Updating model with new vocabulary')
        new_total = pre_exist_total = 0
        # New words and pre-existing words are two separate lists
        new_words = []
        pre_exist_words = []
        if self.nonce is not None:
        #if self.nonce is not None and self.nonce in wv.vocab:
            if self.nonce in wv.vocab:
                gold_nonce = '{}_true'.format(self.nonce)
                nonce_index = wv.vocab[self.nonce].index
                wv.vocab[gold_nonce] = wv.vocab[self.nonce]
                wv.index2word[nonce_index] = gold_nonce
                #del wv.index2word[wv.vocab[self.nonce].index]
                del wv.vocab[self.nonce]
            for word, v in iteritems(self.raw_vocab):
                # Update count of all words already in vocab
                if word in wv.vocab:
                    pre_exist_words.append(word)
                    pre_exist_total += v
                    if not dry_run:
                        wv.vocab[word].count += v
                else:
                    # For new words, keep the ones above the min count
                    # AND the nonce (regardless of count)
                    if keep_vocab_item(word, v, min_count,
                                       trim_rule=trim_rule) or word == self.nonce:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            wv.vocab[word] = Vocab(count=v,
                                                   index=len(wv.index2word))
                            wv.index2word.append(word)
                    else:
                        drop_unique += 1
                        drop_total += v
            original_unique_total = len(pre_exist_words) \
                + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) \
                * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info('New added %i unique words (%i%% of original %i) '
                        'and increased the count of %i pre-existing words '
                        '(%i%% of original %i)', len(new_words),
                        new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct,
                        original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        # Only retaining one subsampling notion from original gensim implementation
        else:
            threshold_count = sample * retain_total

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = wv.vocab[w].count
            word_probability = (np.sqrt(v / threshold_count) + 1) \
                * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info('deleting the raw counts dictionary of %i items',
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info('sample=%g downsamples %i most-common words', sample,
                    downsample_unique)
        logger.info('downsampling leaves estimated %i word corpus '
                    '(%.1f%% of prior %i)', downsample_total,
                    downsample_total * 100.0 / max(retain_total, 1),
                    retain_total)

        # return from each step: words-affected, resulting-corpus-size,
        # extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total,
            'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total),
            'num_retained_words': len(retain_words)
        }

        if self.null_word:
            # create null pseudo-word for padding when using concatenative
            # L1 (run-of-words)
            # this word is only ever input – never predicted – so count,
            # huffman-point, etc doesn't matter
            self.add_null_word(wv)

        if self.sorted_vocab and not update:
            self.sort_vocab(wv)
        if hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree(wv)
        if negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table(wv)

        return report_values, pre_exist_words


class Nonce2VecTrainables(Word2VecTrainables):

    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
        super(Nonce2VecTrainables, self).__init__(vector_size, seed, hashfxn)
        self.info = None

    @classmethod
    def load(cls, w2v_trainables):
        n2v_trainables = cls()
        for key, value in w2v_trainables.__dict__.items():
            setattr(n2v_trainables, key, value)
        return n2v_trainables

    def prepare_weights(self, pre_exist_words, hs, negative, wv, sentences,
                        nonce, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception('prepare_weight on Nonce2VecTrainables should '
                            'always be used with update=True')
        else:
            self.update_weights(pre_exist_words, hs, negative, wv, sentences,
                                nonce)

    def update_weights(self, pre_exist_words, hs, negative, wv, wv_random,
                       nonce):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info('updating layer weights')
        gained_vocab = len(wv.vocab) - len(wv.vectors)
        # newvectors = empty((gained_vocab, wv.vector_size), dtype=REAL)
        newvectors = np.zeros((gained_vocab, wv.vector_size), dtype=np.float32)

        # randomize the remaining words
        # FIXME as-is the code is bug-prone. We actually only want to
        # initialize the vector for the nonce, not for the remaining gained
        # vocab. This implies that the system should be run with the same
        # min_count as the pre-trained background model. Otherwise
        # we won't be able to sum as we won't have vectors for the other
        # gained background words
        if gained_vocab > 1:
            raise Exception('Creating sum vector for non-nonce word. Do '
                            'not specify a min_count when running Nonce2Vec.')
        if gained_vocab == 0:
            raise Exception('Nonce word \'{}\' already in test set and not '
                            'properly deleted'.format(nonce))
        for i in xrange(len(wv.vectors), len(wv.vocab)):
            # Initialise to sum
            for w in pre_exist_words:
               if wv.vocab[w].sample_int > wv_random.rand() * 2**32 or w == nonce:
                   #print "Adding",w,"to initialisation..."
                   newvectors[i-len(wv.vectors)] += wv.vectors[
                       wv.vocab[w].index]


        # Raise an error if an online update is run before initial training on
        # a corpus
        if not len(wv.vectors):
            raise RuntimeError('You cannot do an online vocabulary-update of a '
                               'model which has no prior vocabulary. First '
                               'build the vocabulary of your model with a '
                               'corpus before doing an online update.')

        wv.vectors = np.vstack([wv.vectors, newvectors])
        if negative:
            self.syn1neg = np.vstack([self.syn1neg,
                                         np.zeros((gained_vocab,
                                                      self.layer1_size),
                                                     dtype=np.float32)])
        wv.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = np.ones(len(wv.vocab),
                                        dtype=np.float32)  # zeros suppress learning


class Nonce2Vec(Word2Vec):

    MAX_WORDS_IN_BATCH = 10000

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 callbacks=(), max_final_vocab=None, window_decay=0,
                 sample_decay=1.0):
        super(Nonce2Vec, self).__init__(sentences, size, alpha, window,
                                        min_count, max_vocab_size, sample,
                                        seed, workers, min_alpha, sg, hs,
                                        negative, cbow_mean, hashfxn, iter,
                                        null_word, trim_rule, sorted_vocab,
                                        batch_words, compute_loss, callbacks)
        self.trainables = Nonce2VecTrainables(seed=seed, vector_size=size,
                                              hashfxn=hashfxn)
        #self.lambda_den = 0.0
        self.lambda_den=float(lambda_den)
        self.sample_decay = float(sample_decay)
        self.window_decay = int(window_decay)

    @classmethod
    def load(cls, *args, **kwargs):
        w2vec_model = super(Nonce2Vec, cls).load(*args, **kwargs)
        n2vec_model = cls()
        for key, value in w2vec_model.__dict__.items():
            setattr(n2vec_model, key, value)
        return n2vec_model

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences.
        Return 2-tuple `(effective word count after ignoring unknown words
        and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if not self.sg:
            raise Exception('Nonce2Vec does not support cbow mode')
        logger.info('Training n2v with original code')
        tally += train_batch_sg(self, sentences, alpha, work)
        return tally, self._raw_word_count(sentences)

    def build_vocab(self, sentences, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        total_words, corpus_count = self.vocabulary.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        report_values, pre_exist_words = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, update=update,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(
            vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(pre_exist_words, self.hs,
                                        self.negative, self.wv,
                                        self.random, self.vocabulary.nonce,
                                        update=update)

    def recompute_sample_ints(self):
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))

#####################################################################################

### START OF THE NOVELS VERSION OF N2V

### NOVELS EDIT: added the '_novels' mark to the name of the function
def train_sg_pair_novels(model, word, context_index, alpha, nonce_count,
                  learn_vectors=True, learn_hidden=True, context_vectors=None,
                  context_locks=None, compute_loss=False, is_ft=False):
    if context_vectors is None:
        #context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        #context_locks = model.syn0_lockf
        context_locks = model.trainables.vectors_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)
    l1 = context_vectors[context_index]  # input word (NN input/projection layer)

    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]

        ### NOVELS EDIT: this section has been commented out because in this implementation the alpha is set per-sentence, not per word
        #lambda_den = model.lambda_den
        #exp_decay = -(nonce_count-1) / lambda_den
        #if alpha * np.exp(exp_decay) > model.min_alpha:
            #alpha = alpha * np.exp(exp_decay)
        #else:
            #alpha = model.min_alpha
        logger.debug('training on \'{}\' and \'{}\' with '
                     'alpha = {}'.format(model.vocabulary.nonce,
                                         word,
                                         round(alpha, 5)))
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = np.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += np.outer(gb, l1)
                # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    ### NOVELS EDIT: added the alpha as the second variable returned by the function, so as to reduce it word after word
    return neu1e

### NOVELS EDIT: added the sentence_count argument, needed for keeping track of the current sentence i.e. to make sure the alpha is adjusted accordingly to the current training point. - also added the '_novels' mark to the function called
def train_batch_sg_novels(model, sentences, sentence_count, alpha, work=None, compute_loss=False):
    result = 0
    window = model.window
    ###NOVELS EDIT: added a debug for sample_decay and window decay
    print('Sampling: {}\tWindow: {}\nWindow decay: {}\tSample decay: {}\tLambda: {}'.format(model.sample, window, model.window_decay, model.sample_decay, model.lambda_den))
    # Count the number of times that we see the nonce
    ### NOVELS EDIT: equalized nonce_count and sentence_count, which lets keep track of the current sentence being trained, and adjusted alpha all along the descent through the novel. 
    nonce_count=sentence_count
    #nonce_count = 0
    ### START OF NOVELS EDIT: this is just to calculate the current alpha - in N2V this is done for every word pair, here for every sentence.
    exp_decay = -(nonce_count-1) / model.lambda_den
    if alpha * np.exp(exp_decay) > model.min_alpha:
        alpha = alpha * np.exp(exp_decay)
    else:
        alpha = model.min_alpha
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in
                       model.wv.vocab and model.wv.vocab[w].sample_int
                       #> model.random.rand() * 2 ** 32 or w == '[MASK]']
                       > model.random.rand() * 2 ** 32 or w == model.vocabulary.nonce]
        ### NOVELS_EDIT: added this line to create a list with the subsampled words
        for pos, word in enumerate(word_vocabs):
            # Note: we have got rid of the random window size
            ### NOVELS EDIT: added 'model.' before window
            start = max(0, pos - model.window)
            ### NOVELS EDIT: added this variable in order to keep under control the subsampled line - so it avoids printing out multiple times the same words
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1)],
                                         start):
                # don't train on the `word` itself
                if pos2 != pos:
                    # If training context nonce, increment its count
                    if model.wv.index2word[word2.index] == \
                     model.vocabulary.nonce:
                        ### NOVELS EDIT: commented out the following line, because the nonce_count doesn't need to be updated at this point - it is updated each time we train on a new sentence only
                        #nonce_count += 1
                        ### NOVELS EDIT: printing the nonce count value
                        #print('Nonce count value: {}'.format(nonce_count))
                        #train_sg_pair(
                        #    model, model.wv.index2word[word.index],
                        #    word2.index, alpha, nonce_count,
                        #    compute_loss=compute_loss)
                        ### NOVELS EDIT: had to change the argument word2.index to len(model.wv.vocab)-1, otherwise it would keep asking for an out-of-index value, corresponding to the length of the vocabulary - also added the '_novels' mark to the function called
                        train_sg_pair_novels(
                            model, model.wv.index2word[word.index],
                            len(model.wv.vocab)-1, alpha, nonce_count,
                            compute_loss=compute_loss)
        ###NOVELS_EDIT: Added this line to  print the subsampled line and the current alpha
        sub_line=[model.wv.index2word[word.index] for word in word_vocabs]
        print('Current sentence alpha: {}\nSubsampled line: {}'.format(alpha, sub_line))
        ### END OF NOVELS EDIT
        result += len(word_vocabs)
            ###  NOVELS EDIT: added 'model.' before calling window, trying to see if this way it works
        if model.window - 1 >= 3:
            ###  NOVELS EDIT: added 'model.' before calling window, trying to see if this way it works
            model.window = window - model.window_decay
        ### NOVELS EDIT: commented out this line, and added an easy fix to the subsampling issue
        #model.recompute_sample_ints()
        #model.sample = model.sample / model.sample_decay
    return result

### NOVELS EDIT: added the '_novels' mark to the class name
class Nonce2VecVocab_novels(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
### NOVELS EDIT: added the '_novels' mark to the class name
        super(Nonce2VecVocab_novels, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    @classmethod
    def load(cls, w2v_vocab):
        """Load a Nonce2VecVocab instance from a Word2VecVocab instance."""
        n2v_vocab = cls()
        for key, value in w2v_vocab.__dict__.items():
            setattr(n2v_vocab, key, value)
        return n2v_vocab

    ### NOVELS EDIT: added sentence_count to the arguments, which is passed from the function in main.py and allows to keep track of the number of sentences we have trained on so far
    def prepare_vocab(self, sentence_count, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            raise Exception('Nonce2Vec can only update a pre-existing '
                            'vocabulary')
        logger.info('Updating model with new vocabulary')
        new_total = pre_exist_total = 0
        # New words and pre-existing words are two separate lists
        new_words = []
        pre_exist_words = []
        if self.nonce is not None:
            ### NOVELS EDIT: added this condition, which makes sure that the character's vector is reinitialized to zero only once, during the first sentence.
            if sentence_count==1:
            #if self.nonce is not None and self.nonce in wv.vocab:
                if self.nonce in wv.vocab:
                    print('Sentence count == 1 and nonce deleted')
                    original_nonce = '{}_original'.format(self.nonce)
                    nonce_index = wv.vocab[self.nonce].index
                    wv.vocab[original_nonce] = wv.vocab[self.nonce]
                    wv.index2word[nonce_index] = original_nonce
                    #del wv.index2word[wv.vocab[self.nonce].index]
                    del wv.vocab[self.nonce]
                    if self.nonce not in wv.vocab:
                        logger.info('Deleted the vector for the nonce - index no. {}'.format(nonce_index))
                self.nonce = '[MASK]'
                    
            for word, v in iteritems(self.raw_vocab):
                # Update count of all words already in vocab
                if word in wv.vocab:
                    pre_exist_words.append(word)
                    pre_exist_total += v
                    if not dry_run:
                        wv.vocab[word].count += v
                else:
                    # For new words, keep the ones above the min count
                    # AND the nonce (regardless of count)
                    ### NOVELS EDIT: commented out this condition, added another one which keeps only the nonce
                    #if keep_vocab_item(word, v, min_count,
                    #                   trim_rule=trim_rule) or word == self.nonce:
                    if word == self.nonce and word not in new_words:
                        print('Added this word to the list of unknown words: {}'.format(word))
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            wv.vocab[word] = Vocab(count=v,
                                                   index=len(wv.index2word))
                            wv.index2word.append(word)
                    else:
                        drop_unique += 1
                        drop_total += v
            original_unique_total = len(pre_exist_words) \
                + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) \
                * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info('New added %i unique words (%i%% of original %i) '
                        'and increased the count of %i pre-existing words '
                        '(%i%% of original %i)', len(new_words),
                        new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct,
                        original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        # Only retaining one subsampling notion from original gensim implementation
        else:
            threshold_count = sample * retain_total

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = wv.vocab[w].count
            word_probability = (np.sqrt(v / threshold_count) + 1) \
                * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info('deleting the raw counts dictionary of %i items',
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info('sample=%g downsamples %i most-common words', sample,
                    downsample_unique)
        logger.info('downsampling leaves estimated %i word corpus '
                    '(%.1f%% of prior %i)', downsample_total,
                    downsample_total * 100.0 / max(retain_total, 1),
                    retain_total)

        # return from each step: words-affected, resulting-corpus-size,
        # extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total,
            'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total),
            'num_retained_words': len(retain_words)
        }

        if self.null_word:
            # create null pseudo-word for padding when using concatenative
            # L1 (run-of-words)
            # this word is only ever input – never predicted – so count,
            # huffman-point, etc doesn't matter
            self.add_null_word(wv)

        if self.sorted_vocab and not update:
            self.sort_vocab(wv)
        if hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree(wv)
        if negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table(wv)

        return report_values, pre_exist_words

### NOVELS EDIT: added the '_novels' mark at the end of the class name
class Nonce2VecTrainables_novels(Word2VecTrainables):

    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
### NOVELS EDIT: added the '_novels' mark at the end of the class name
        super(Nonce2VecTrainables_novels, self).__init__(vector_size, seed, hashfxn)
        self.info = None

    @classmethod
    def load(cls, w2v_trainables):
        n2v_trainables = cls()
        for key, value in w2v_trainables.__dict__.items():
            setattr(n2v_trainables, key, value)
        return n2v_trainables

    def prepare_weights(self, pre_exist_words, hs, negative, wv, sentences,
                        nonce, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception('prepare_weight on Nonce2VecTrainables should '
                            'always be used with update=True')
        else:
            self.update_weights(pre_exist_words, hs, negative, wv, sentences,
                                nonce)

    def update_weights(self, pre_exist_words, hs, negative, wv, wv_random,
                       nonce):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info('updating layer weights - current nonce: {}'.format(nonce))
        gained_vocab = len(wv.vocab) - len(wv.vectors)
        # newvectors = empty((gained_vocab, wv.vector_size), dtype=REAL)
        newvectors = np.zeros((gained_vocab, wv.vector_size), dtype=np.float32)

        # randomize the remaining words
        # FIXME as-is the code is bug-prone. We actually only want to
        # initialize the vector for the nonce, not for the remaining gained
        # vocab. This implies that the system should be run with the same
        # min_count as the pre-trained background model. Otherwise
        # we won't be able to sum as we won't have vectors for the other
        # gained background words
        if gained_vocab > 1:
            raise Exception('Creating sum vector for non-nonce word. Do '
                            'not specify a min_count when running Nonce2Vec.')
        ### NOVELS EDIT: had to comment out the next exception, because the first sentence, in the novels setting gained_vocab should be precisely 0.
        '''if gained_vocab == 0:
            raise Exception('Nonce word \'{}\' already in test set and not '
                            'properly deleted'.format(nonce))'''
        for i in xrange(len(wv.vectors), len(wv.vocab)):
            # Initialise to sum
            for w in pre_exist_words:
                ### NOVELS EDIT: rmoved the following condition, added a simpler one, which btw avoids adding the disgusting vector for '[MASK]'
                if wv.vocab[w].sample_int > wv_random.rand() * 2**32 and w != nonce:
                   ### NOVELS EDIT: modified the print text, does the same thing
                   #print('Adding {} to initialisation...'.format(w))
                   newvectors[i-len(wv.vectors)] += wv.vectors[
                       wv.vocab[w].index]

        # Raise an error if an online update is run before initial training on
        # a corpus
        if not len(wv.vectors):
            raise RuntimeError('You cannot do an online vocabulary-update of a '
                               'model which has no prior vocabulary. First '
                               'build the vocabulary of your model with a '
                               'corpus before doing an online update.')

        wv.vectors = np.vstack([wv.vectors, newvectors])
        if negative:
            self.syn1neg = np.vstack([self.syn1neg,
                                         np.zeros((gained_vocab,
                                                      self.layer1_size),
                                                     dtype=np.float32)])
        wv.vectors_norm = None

        ### CHECK HERE ###
        # do not suppress learning for already learned words
        self.vectors_lockf = np.ones(len(wv.vocab),
                                        dtype=np.float32)  # zeros suppress learning

### NOVELS EDIT: added the '_novels' mark to the class name
class Nonce2Vec_novels(Word2Vec):

    MAX_WORDS_IN_BATCH = 10000

    def __init__(self, sentences=None, sentence_count=0, size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 callbacks=(), max_final_vocab=None, window_decay=0,
                 sample_decay=1.0):
        ### NOVELS EDIT: added the '_novels' mark to the class name
        super(Nonce2Vec_novels, self).__init__(sentences, size, alpha, window,
                                        min_count, max_vocab_size, sample,
                                        seed, workers, min_alpha, sg, hs,
                                        negative, cbow_mean, hashfxn, iter,
                                        null_word, trim_rule, sorted_vocab,
                                        batch_words, compute_loss, callbacks)
        ### NOVELS EDIT: added the '_novels' mark to the class name
        self.trainables = Nonce2VecTrainables_novels(seed=seed, vector_size=size,
                                              hashfxn=hashfxn)
        self.lambda_den = 0.0
        self.sample_decay = float(sample_decay)
        self.window_decay = int(window_decay)
        self.window=int(window)
        ### NOVEL EDIT: added the self.sentence_count to make sure it initializes to the passed value        
        self.sentence_count = int(sentence_count)

    @classmethod
    def load(cls, *args, **kwargs):
        ### NOVELS EDIT: added the '_novels' mark to the called class
        w2vec_model = super(Nonce2Vec_novels, cls).load(*args, **kwargs)
        n2vec_model = cls()
        for key, value in w2vec_model.__dict__.items():
            setattr(n2vec_model, key, value)
        return n2vec_model

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences.
        Return 2-tuple `(effective word count after ignoring unknown words
        and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if not self.sg:
            raise Exception('Nonce2Vec does not support cbow mode')
        logger.info('Training n2v with original code')
        ### NOVELS EDIT: added the sentence_count, which allows to keep track of the increase in number of the times we see each character, by increasing by 1 per sentence. 
        sentence_count=self.sentence_count
        ### NOVELS EDIT: added the sentence_count argument to the train_batch_sg function, in order to consider it when training - and added the '_novels' mark to the function called
        tally += train_batch_sg_novels(self, sentences, sentence_count, alpha, work) 
        return tally, self._raw_word_count(sentences)

    ### NOVELS EDIT: added the sentence_count argument, 
    def build_vocab(self, sentences, sentence_count, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        total_words, corpus_count = self.vocabulary.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        ### NOVELS EDIT: added the sentence_count argument, needed in order not to delete the character vector every time we see a new sentence i.e. every time we take another training step
        report_values, pre_exist_words = self.vocabulary.prepare_vocab(
            sentence_count, self.hs, self.negative, self.wv, update=update,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(
            vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(pre_exist_words, self.hs,
                                        self.negative, self.wv,
                                        self.random, self.vocabulary.nonce,
                                        update=update)

    def recompute_sample_ints(self):
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))
