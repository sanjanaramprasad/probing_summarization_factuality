import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
import torch.nn.functional as F
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
import argparse
import numpy as np
import string
import difflib
import string
import re 
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scripts.utils import load_model, longest_common_substring_ignore_punctuations

nltk.download('stopwords')
stop_words = stopwords.words('english')
hf_logging.set_verbosity_error()

device = 'cuda'


class ModelStateDataset():
    def __init__(self, 
                 model_name, 
                 doc_truncate = 2000, 
                 summary_truncate = 2000):
        
        self.model_name = model_name
        tokenizer, model = load_model(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id == None else tokenizer.pad_token_id
        self.model = model
        self.tokenizer = tokenizer
        #### Do not use layer 0 ###
        self.doc_truncate = doc_truncate
        self.summary_truncate = summary_truncate

    def make_prompt_ids(self, 
                       prompt_prefix,
                       prompt_suffix):

        
        self.prefix_ids = self.tokenizer(prompt_prefix, return_tensors="pt").input_ids if prompt_prefix else None
        self.suffix_ids = None
        if prompt_suffix:
            suffix_ids = self.tokenizer(prompt_suffix, return_tensors="pt").input_ids
            suffix_ids = suffix_ids[:, 1:] if suffix_ids[0][0] in [0,1,2]  else suffix_ids
            self.suffix_ids = suffix_ids
        
        return

    def update_label_pos(self,
                         summary,
                         inconsistent_spans):
        labels = [0] * len(summary.split(' '))
        for nonfactual_span in inconsistent_spans:
            processed_nonfactual_span = longest_common_substring_ignore_punctuations(nonfactual_span, summary)
        
            if processed_nonfactual_span != None and re.search(processed_nonfactual_span, summary):
                start_idx, end_idx = re.search(processed_nonfactual_span, summary).span()
                curr_char_idx = 0
            
                for widx, w in enumerate(summary.split(' ')):
                    end_char_idx = curr_char_idx + (len(w) - 1 )
                    assert (summary[curr_char_idx: end_char_idx + 1] == w)
                    if curr_char_idx>= start_idx and end_char_idx <= end_idx:
                        labels[widx] = 1
                    curr_char_idx = end_char_idx + 2
            else:
                print('ERR', nonfactual_span, '|', summary)
                print('***'* 13)
        return labels

    def get_tokens_labels(self,
                         inconsistent_spans, 
                         summary):
        
        labels = self.update_label_pos(summary,
                                  inconsistent_spans)
                
        words_labels = list(zip(summary.split(' '), labels))
    
        summ_tokens = []
        summ_tokens_labels = []
        word_count = 0
        for w, l in words_labels:
            if self.model_name == 'falcon7b':
                w = f' {w}'
                # print('WT', self.tokenizer(f'{w}').input_ids, self.tokenizer(f' {w}').input_ids)
            word_tokens = self.tokenizer(w).input_ids
            word_tokens = word_tokens[1:] if word_tokens[0] in [0,1,2] else word_tokens
            summ_tokens += word_tokens
            summ_tokens_labels += [l] * len(word_tokens)
            word_count += 1

        assert(len(summ_tokens) == len(summ_tokens_labels))
        return summ_tokens, summ_tokens_labels

    def get_hstates_prediction(self, hstate):
        logits = self.model.lm_head(hstate.to('cuda'))
        probs = F.softmax(logits, dim=-1).cpu().detach()
        predicted_tokens = []
        for layer in probs:
            predicted_tokens.append(torch.tensor([torch.argmax(layer).item()]))
        assert(len(predicted_tokens) == probs.shape[0])
        predicted_tokens = torch.stack(predicted_tokens, dim = 0)
        return predicted_tokens, probs.cpu().detach().numpy()

    def make_truncated_prompt_tokens_labels(self,
                                            doc,
                                            summary,
                                            inconsistent_spans,
                                            padding = True
                                           ):
       
        
        tokens_labels_dict = {}
        # print(doc)
        doc_ids = self.tokenizer(doc, return_tensors="pt").input_ids
        doc_ids = doc_ids[:,1:] if doc_ids[0][0] in [0,1,2] else doc_ids
        summ_tokens, summary_token_labels = self.get_tokens_labels(inconsistent_spans, summary)

        ####truncate all #####
        # doc_ids = doc_ids[:,:self.doc_truncate]
        # summ_tokens = summ_tokens[:self.summary_truncate]
        # summary_token_labels = summary_token_labels[:self.summary_truncate]
        prompt_ids = torch.tensor([])
        if self.prefix_ids != None:
            prompt_ids = torch.cat([self.prefix_ids, doc_ids], dim = -1)
        if self.suffix_ids != None:
            prompt_ids = torch.cat([prompt_ids, self.suffix_ids], dim = -1)
        
        if prompt_ids.shape[-1] != 0:
            all_tokens = torch.cat([prompt_ids.squeeze(0), torch.tensor(summ_tokens)], dim = -1)
        else:
            all_tokens = torch.tensor(summ_tokens)
        
        # print(all_tokens.tolist())
        # print('PROMPT', self.tokenizer.decode(all_tokens.tolist()))
        # print('***'* 13)
        if padding:
            max_rows = self.doc_truncate + self.summary_truncate
            pad_rows = max_rows - all_tokens.shape[-1]
            all_tokens = all_tokens.tolist() + [self.tokenizer.pad_token_id] * pad_rows
            all_tokens = torch.tensor(all_tokens)
            
            max_summ_tokens = self.summary_truncate 
            pad_tokens = max_summ_tokens - len(summary_token_labels)
            summary_token_labels = summary_token_labels + [-100] * pad_tokens
            summary_token_labels = torch.tensor(summary_token_labels)

        tokens_labels_dict['all_tokens'] = all_tokens
        tokens_labels_dict['prompt_ids'] = prompt_ids 
        tokens_labels_dict['summ_tokens'] = summ_tokens
        tokens_labels_dict['summary_token_labels'] = summary_token_labels
        
        
        return tokens_labels_dict
        
    def get_internal_states(self,
                            doc,
                            summary,
                            nonfactual_spans,
                            prompt_type,
                            prompt_template_path,
                            padding = False):
        example_dict = {}
        prompt_file = f'{prompt_template_path}/{prompt_type}.json'
        with open(prompt_file, 'r') as fp:
            prompt_doc = json.load(fp)
        
        prompt_prefix = prompt_doc['prompt_prefix_template']
        prompt_suffix = prompt_doc['prompt_suffix_template']
        
        self.make_prompt_ids(prompt_prefix,
                            prompt_suffix)

        inconsistent_spans = nonfactual_spans.split('<sep>') if type(nonfactual_spans) is str else []

        tokens_labels_dict = self.make_truncated_prompt_tokens_labels(doc,
                                            summary,
                                            inconsistent_spans,
                                            padding = padding)
        # print('INP SHAPE', tokens_labels_dict['all_tokens'].unsqueeze(0).shape)
        with torch.no_grad():
            outputs = self.model.generate(tokens_labels_dict['all_tokens'].unsqueeze(0).to(device), 
                        max_length=tokens_labels_dict['all_tokens'].shape[0] + 1, 
                        output_attentions = True,
                        output_hidden_states=True, 
                        return_dict_in_generate=True)
            
        
        source_len = tokens_labels_dict['prompt_ids'].shape[-1]
        summary_len = len(tokens_labels_dict['summ_tokens'])
        output_hidden_states = outputs['hidden_states'][0]
        output_attentions = outputs['attentions'][0]
        output_attentions = [each.detach().cpu() for each in output_attentions]
        del outputs
        # print(len(output_attentions), output_attentions[0].shape, type(output_attentions))
        # output_layer_attns = [each[:, source_len - 1: source_len + summary_len - 1, :, :] for each in output_attentions]
        # # summary_hidden_states = output_attentions[:]
        example_dict['all_tokens'] = tokens_labels_dict['all_tokens']
        example_dict['summary_token_labels'] = tokens_labels_dict['summary_token_labels']
        example_dict['source_len'] = torch.tensor([source_len])
        example_dict['summary_len'] = torch.tensor([summary_len])
        example_dict['hidden_states'] = torch.cat(output_hidden_states).cpu()
        example_dict['attentions'] = torch.cat(output_attentions)
        
        # print(example_dict['attentions'].shape, len(example_dict['summary_token_labels']), example_dict['summary_len'])
        # del outputs
        return example_dict



def run_test_internal_states(doc,
                             summary,
                             results, 
                             model_state,
                             print_generation_ids):
    all_tokens = results['all_tokens']
    source_len = results['source_len'][0]
    summary_len = results['summary_len'][0]
    summary_token_labels = results['summary_token_labels']
    hidden_states = results['hidden_states']
    attentions = results['attentions']

    # print(len(all_tokens))
    tokenized_doc = model_state.tokenizer.decode(all_tokens[:source_len])
    tokenized_doc = tokenized_doc.split('\nDocument:')[-1].split('\nSummary:')[0].strip()
    tokenized_summary = model_state.tokenizer.decode(all_tokens[source_len: source_len + summary_len])

    if not(tokenized_summary.strip() == summary.strip()):
        print(summary)
        print('--'* 13)
        print(tokenized_summary)
        print('***'* 13)

    # summ
    if print_generation_ids:
        summary_hidden_states = hidden_states[:, source_len - 1: source_len + summary_len - 1, ]
        summary_tokens = all_tokens[source_len : source_len + summary_len]
        # print(summary_hidden_states.shape, len(summary_tokens), summary_len)
        for tok_idx, tok in enumerate(summary_tokens):
            tok_hstate = summary_hidden_states[:, tok_idx, :]
            # print(tok_hstate.shape)
            predicted_tokens, probs = model_state.get_hstates_prediction(tok_hstate)
            # print(predicted_tokens, tok)
            print(model_state.tokenizer.decode([predicted_tokens[-1][0]]), model_state.tokenizer.decode([tok]) )
    return


'''
prompts 
1) Document + Context: 
instruction: Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document. 
prompt_template : {{instruction}}\nDocument: {{doc}}\nSummary: {{summary}}
2) Context alone 
prompt template: {{summary_prefx}}
3) context + topic estimation 
instructon: Complete the following document that talks about topic {{topic}} 
prompt template: {{instructon}}\n{{summary_prefix}}
4) Mask summary entities out 

'''
def write_data(df,
               source_key,
               summary_key,
               nonfactual_span_key,
               model_state,
               model, 
               origin,
               prompt_type,
               prompt_template_path,
               write_path):
    df = df[~df[source_key].isnull()]
    counter = -1
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        counter += 1
        try:
            doc = row[source_key]
            summary = row[summary_key]
            nonfactual_spans = row[nonfactual_span_key]
            results = model_state.get_internal_states(doc, 
                                    summary,
                                    nonfactual_spans,
                                    prompt_type = prompt_type,
                                    prompt_template_path = prompt_template_path)
            if counter == 0:
                print_generation_ids = True
            else:
                print_generation_ids = False

            run_test_internal_states(doc,
                                    summary,
                                    results, 
                                    model_state,
                                    print_generation_ids = print_generation_ids
                                    )
            doc_name = row['id'] if 'id' in row else idx

            torch.save(results, f'{write_path}/{origin}/{model}/{prompt_type}/{doc_name}.pt')
        except Exception as e:
            print(e, idx)
            continue
        # break
        
    return
        
    
def write_states(args,
                instruction):
    model = args.model
    origin = args.origin 
    read_path = args.read_path 
    source_key = args.source_key
    summary_key = args.summary_key
    nonfactual_span_key = args.nonfactual_span_key
    write_path = args.write_path
    prompt_template_path =  args.prompt_template_path  
    prompt_type = args.prompt_type 
    
    
    df = pd.read_csv(read_path)
    if 'origin' in df:
        df = df[df['origin'] == origin]
    
    df = df[df['model'] == model]
    
    print(f'Storing internal states for {model} on summaries of {origin}...')
    model_state = ModelStateDataset(model)
    write_data(df,
               source_key,
               summary_key,
               nonfactual_span_key,
               model_state,
               model,
               origin,
               prompt_type,
               prompt_template_path,
               write_path)
               
    

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-origin", 
                           "--origin",
                          default = 'XSUM')
    
    argParser.add_argument("-model", 
                           "--model",
                          default = ' ')

    argParser.add_argument("-source_key", 
                           "--source_key",
                          default = 'source')
    
    argParser.add_argument("-summary_key", 
                           "--summary_key",
                          default = 'summary')
    
    argParser.add_argument("-nonfactual_span_key", 
                           "--nonfactual_span_key",
                          default = 'annotated_spans')
    
    argParser.add_argument("-prompt_type", 
                           "--prompt_type",
                          default = 'document_context')
    
    argParser.add_argument("-prompt_template_path", 
                           "--prompt_template_path",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/prompt_templates/')
    
    argParser.add_argument("-read_path", 
                           "--read_path",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/Genaudit_annotations.csv')
    
    argParser.add_argument("-write_path", 
                           "--write_path",
                          default = '/scratch/ramprasad.sa/probing_summarization_factuality/internal_states/Genaudit')

    

    

    args = argParser.parse_args()

    instruction = "Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document."
    
    write_states(args,
                instruction)
    print('Done!')

