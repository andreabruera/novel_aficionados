### EXAMPLE: python3 -m scripts.bert ./baskerville.txt 4

import collections
import logging
import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nonce2vec.utils.count_based_models_utils import cosine_similarity
from torch.nn import CosineSimilarity, PairwiseDistance

#logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)

#parser = argparse.ArgumentParser()
#parser.add_argument('filepath', type = str, help = 'Absolute path to file')
#parser.add_argument('layers', type = int, help = 'Number of layers of BERT to be extracted')
#args = parser.parse_args()

class BERT_test:
        
    def train(args, sentence, character, model, tokenizer):
        layers = args.bert_layers
    #with open(args.filepath) as sample:
       
        #sample = sample.readlines()
        
        #halves = [sample[:int(((len(sample)) / 2) - 1)], sample[int((len(sample)) / 2):]]
        
        #for half_index, half in enumerate(halves):
        masked_index = [i for i, v in enumerate(sentence) if v == '[MASK]'][0]
        for word_index, word in enumerate(sentence): 
            if word == '[MASK]' and word_index != masked_index:
                del sentence[word_index]
        line = ' '.join(sentence)  
        text = "[CLS] {}".format(line)
        tokenized_text = tokenizer.tokenize(text)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        model.eval()
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)

        summed_vector = [word_tensor[0][masked_index] for word_tensor in encoded_layers] 
      
        #for layer_index, layer in enumerate(encoded_layers):
            #if layer_index == 0:
                #summed_vector = layer[0][masked_index]
            #else:
                #summed_vector += layer[0][masked_index]

        return summed_vector
