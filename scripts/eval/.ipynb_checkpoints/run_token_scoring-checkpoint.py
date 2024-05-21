import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from scripts.dataset_creators.read_internal_states import HiddenStatesDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = {'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.1',
             'falcon7b': 'tiiuae/falcon-7b-instruct',
              'llama7b': '/work/frink/models/Llama-2-7b-chat-hf',
             'flanul2': 'google/flan-ul2'}

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],
                                         cache_dir = '/scratch/ramprasad.sa/huggingface_models')
    return tokenizer



def is_new_word(token, prev_token, tokenizer):
    tokenized_str = tokenizer.decode([token])
    if prev_token != None:
        tokenized_str = tokenizer.decode([prev_token, token])                
    # print(tokenized_str)
    if len(tokenized_str.split(' ')) != len(tokenizer.decode([token]).split(' ')):
                return True
    return False

def get_balanced_acc(info ):
    predictions = [each[0] for each in info]
    labels = [each[1] for each in info]
    # balanced_accuracy_score(y_true, y_pred)
    acc = balanced_accuracy_score(labels, predictions)
    return acc

def get_subword_level(tokens,
                    predictions,
                    labels,
                    model_name):
    postprocess_tokens = []
    postprocess_predictions = []
    postprocess_labels = []
    
    subwords = []
    subword_predictions = []
    subword_labels = []
    

    tokenizer = load_tokenizer(model_name)
    for tok_idx, tok in enumerate(tokens):
        pred = predictions[tok_idx]
        lab = labels[tok_idx]
        
        if tok_idx == 0:
            prev_tok = None
        else:
            prev_tok = tokens[tok_idx - 1]

        new_word = is_new_word(tok, prev_tok, tokenizer)
            
        if new_word:
            if subword_predictions:
                postprocess_predictions.append(subword_predictions)
                postprocess_labels.append(subword_labels)
                postprocess_tokens.append(tokenizer.decode(subwords))
                # print(subword_predictions)
            subword_predictions = []
            subword_labels = []
            subwords = []

        subword_predictions += [pred]
        subword_labels += [lab]
        subwords += [tok]
    return postprocess_tokens, postprocess_predictions, postprocess_labels



def postprocess_predictions_labels(tokens, 
                                   predictions, 
                                   labels,
                                   model_name):
  
    
    postprocess_tokens, postprocess_predictions, postprocess_labels = get_subword_level(tokens,
                                                                                        predictions,
                                                                                        labels,
                                                                                        model_name)
    
    scores_token = []
    scores_beg_word = []
    scores_maxpool = []
     
    for pred_subwords, lab_subwords in list(zip(postprocess_predictions, postprocess_labels)):
        assert(len(pred_subwords) == len(lab_subwords))
        scores_token += [(pred, lab) for pred, lab in list(zip(pred_subwords, lab_subwords))]
        scores_beg_word += [(pred_subwords[0], lab_subwords[0])]
        pred = 1 if 1 in pred_subwords else 0 
        lab = 1 if 1 in lab_subwords else 0
        if len(set(lab_subwords)) != 1:
            print('oop')
        scores_maxpool += [(pred, lab)]

    scores = {'token_bacc': get_balanced_acc(scores_token),
              'beg_word_bacc': get_balanced_acc(scores_beg_word),
              'maxpool_bacc': get_balanced_acc(scores_maxpool)}
    return scores