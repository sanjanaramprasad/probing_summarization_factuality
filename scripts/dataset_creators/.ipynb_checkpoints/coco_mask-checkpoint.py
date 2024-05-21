import os
import argparse
import torch
import numpy as np
import spacy
import json
import pandas as pd
from transformers import logging as hf_logging
import torch.nn.functional as F
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
from random import sample
from numpy import argmax
hf_logging.set_verbosity_error()


models = ['falcon7b', 'mistral7b', 'flanul2']
origins = ['ACIBENCH', 'XSUM']

model_path = {'mistral7b': 'mistralai/Mistral-7B-v0.1',
             'falcon7b': 'tiiuae/falcon-7b-instruct',
             'flanul2': 'google/flan-ul2'}

class CocoTokenizer(object):

    def __init__(self, model='en_core_web_sm', use_gpu=True):

        super(CocoTokenizer, self).__init__()
        if use_gpu:
            spacy.prefer_gpu()
        self.nlp_model = spacy.load(model)

    def tokenize_and_pos(self, text):

        result = self.nlp_model(text)
        tokens = [x.text for x in result]
        pos_tags = [x.pos_ for x in result]
        whitespaces = [x.whitespace_ for x in result]
        return tokens, pos_tags, whitespaces

    def sentencizer(self, doc):

        result = self.nlp_model(doc)
        return [sent.text for sent in result.sents]
    
class Coco:
    
    def __init__(self):
        self.universal_pos_tags = ['ADJ','ADP','ADV', 'AUX', 'CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']
        
        self.unimportant_pos_tags = ['PUNCT', 'SYM', 'DET', 'PART', 'CCONJ', 'SCONJ']
        self.important_pos_tags = [tag for tag in self.universal_pos_tags if tag not in self.unimportant_pos_tags]
        
        self.tokenizer = CocoTokenizer()
        
    def get_masked_token_list(self, generated_summ):
        summ_tokens, summ_tags, summ_spaces = self.tokenizer.tokenize_and_pos(generated_summ)
        
        masked_token_list = [k for k,v in zip(summ_tokens, summ_tags) if v in self.important_pos_tags]
        print('DIFF in masked', set(summ_tokens) - set(masked_token_list))
        return masked_token_list
    
    def mask_document(self, source_doc, 
             masked_token_list, 
             mask_token='[MASK]', 
             mask_strategy='token'):

        tokenized_doc, tokenized_doc_pos_tags, tokenized_doc_whitespaces = self.tokenizer.tokenize_and_pos(source_doc)
        masked_token_list = [x.lower() for x in masked_token_list]
        mask_matrix = np.ones_like(tokenized_doc, dtype=np.int32)
        
        if mask_strategy == 'doc':
            mask_matrix = np.zeros_like(tokenized_doc, dtype=np.int32)
            
        elif mask_strategy == 'token':
            for idx,word in enumerate(tokenized_doc):
                if word.lower() in masked_token_list:
                    mask_matrix[idx] = 0
                    
        elif mask_strategy == 'span':
            for idx, word in enumerate(tokenized_doc):
                if word.lower() in masked_token_list:
                    mask_matrix[idx] = 0
                    if idx-1 >= 0:
                        mask_matrix[idx-1] = 0
                    if idx-2 >= 0:
                        mask_matrix[idx-2] = 0
                    if idx+1 < len(tokenized_doc):
                        mask_matrix[idx+1] = 0
                    if idx+2 < len(tokenized_doc):
                        mask_matrix[idx+2] = 0
                        
        elif mask_strategy == 'sent':
            sents = self.tokenizer.sentencizer(source_doc)
            mask_matrix = []
            for sent in sents:
                token_sent = self.tokenizer.tokenize_and_pos(sent)[0]
                token_sent = [x.lower() for x in token_sent]
                sent_mask_matrix = np.ones_like(token_sent, dtype=np.int32)
                for masked_word in masked_token_list:
                    if masked_word in token_sent:
                        sent_mask_matrix = np.zeros_like(token_sent, dtype=np.int32)
                        break
                mask_matrix.append(sent_mask_matrix)
            mask_matrix = np.concatenate(mask_matrix, axis=0)

        assert len(tokenized_doc) == len(mask_matrix)
        masked_doc = np.where(mask_matrix.astype(bool), tokenized_doc, [mask_token]*len(tokenized_doc))
        # print(mask_matrix, tokenized_doc, [mask_token], len(tokenized_doc))
        masked_doc = [tok + tokenized_doc_whitespaces[tok_idx] for tok_idx, tok in enumerate(masked_doc)]
        
        return ''.join(masked_doc)
    
        