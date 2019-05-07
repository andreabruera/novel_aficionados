### EXAMPLE: python3 -m scripts.bert ./baskerville.txt 4

import collections
import logging
import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nonce2vec.utils.count_based_models_utils import cosine_similarity
from torch.nn import CosineSimilarity, PairwiseDistance

logging.basicConfig(format='%(asctime)s - %(message)s', level = logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('filepath', type = str, help = 'Absolute path to file')
parser.add_argument('layers', type = int, help = 'Number of layers of BERT to be extracted')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_representations = collections.defaultdict(torch.Tensor)

with open(args.filepath) as sample:
   
    sample = sample.readlines()
    
    halves = [sample[:int(((len(sample)) / 2) - 1)], sample[int((len(sample)) / 2):]]
    
    model = BertModel.from_pretrained('bert-base-uncased')

    for half_index, half in enumerate(halves):

        training_counter = 0

        first_sentence = True
       
        if half_index == 0:
            character = 'watson'
        else:
            character = 'holmes'

        for line in half:
            
            if character in line and training_counter <= 50:

                training_counter += 1
                text = "[CLS] {}".format(line)
                tokenized_text = tokenizer.tokenize(text)

                if character in tokenized_text:

                    masked_index = [i for i, v in enumerate(tokenized_text) if v == character][0]
                    tokenized_text[masked_index] = '[MASK]'
                    
                    for word_index, word in enumerate(tokenized_text): 
                        if word == character:
                            del tokenized_text[word_index]

                    logging.info('{}'.format(tokenized_text))

                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    model.eval()
                    with torch.no_grad():
                        encoded_layers, _ = model(tokens_tensor)
                  
                    for layer_index, layer in enumerate(encoded_layers[: args.layers]):
                        if layer_index == 0:
                            summed_vector = layer[0][masked_index]
                        else:
                            summed_vector += layer[0][masked_index]

                    if first_sentence == True: 
                        bert_representations[half_index + 1] = summed_vector
                        first_sentence = False
                    else:
                        bert_representations[half_index + 1] += summed_vector

first_part = bert_representations[1].numpy()
second_part = bert_representations[2].numpy()
logging.info(cosine_similarity(first_part, second_part))

