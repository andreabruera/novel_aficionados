### EXAMPLE: python3 -m scripts.bert ./baskerville.txt 4

import collections
import logging
import argparse
import torch
import argparse
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nonce2vec.utils.count_based_models_utils import cosine_similarity
from torch.nn import CosineSimilarity, PairwiseDistance
from collections import defaultdict

#logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)

#parser = argparse.ArgumentParser()
#parser.add_argument('filepath', type = str, help = 'Absolute path to file')
#parser.add_argument('layers', type = int, help = 'Number of layers of BERT to be extracted')
#args = parser.parse_args()

class BERT_test:
        
    def train(args, sentence, character, model, tokenizer):
        layers = args.bert_layers
        sentence.insert(0, '[CLS]')
        temporary_masked_index = [i for i, v in enumerate(sentence) if v == '[MASK]'][0]
        for word_index, word in enumerate(sentence): 
            if word == '[MASK]' and word_index != temporary_masked_index:
                del sentence[word_index]
        
        sentence = ' '.join(sentence)  
        tokenized_text = tokenizer.tokenize(sentence)
        masked_index = [i for i, v in enumerate(tokenized_text) if v == '[MASK]'][0]
        assert tokenized_text[masked_index] == '[MASK]' 
        other_indexes={token : index for index, token in enumerate(tokenized_text) if index>0}

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        model.eval()
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
  

        if args.sum_CLS:
            layered_tensor = [word_tensor[0][0] for word_tensor in encoded_layers] 

        elif args.sum_only:
            summed_layer = defaultdict(torch.Tensor)
            layered_tensor = []
            full_tensor = [word_tensor[0] for word_tensor in encoded_layers]
            for layer in full_tensor:
                for word_index, word_vector in enumerate(layer):
                    if word_index != 0 and word_index != masked_index:
                        if len(summed_layer) == 0:
                            summed_layer = word_vector
                        elif word_index != masked_index:
                            summed_layer += word_vector
                layered_tensor.append(summed_layer)
            
        else:
            layered_tensor = [word_tensor[0][masked_index] for word_tensor in encoded_layers] 
        
        other_words_vectors = {other_word: encoded_layers[11][0][other_word_index] for other_word, other_word_index in other_indexes.items()}

        assert len(layered_tensor) == 12
        layered_tensor = torch.stack(layered_tensor)

        return layered_tensor, other_words_vectors
