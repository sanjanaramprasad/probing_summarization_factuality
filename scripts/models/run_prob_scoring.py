import torch
import os 
import uuid
import pandas as pd
import numpy as np
import argparse
import torch.nn.functional as F
from scripts.utils import load_model, get_l1, get_document_keywords, longest_common_substring_ignore_punctuations
from tqdm import tqdm
from scipy.stats import entropy
from scripts.dataset_creators.coco_mask import Coco
import json
import re

device = 'cuda'

def strip_bos_token(ids):
    if ids[0][0] in [0, 1, 2]:
        ids = ids[:, 1:]
    return ids

def strip_eos_token(ids):
    if ids[0][-1] in [0, 1, 2, 11]:
        ids = ids[:, :-1]
    return ids


device = 'cuda'

class Benchmarks:
    
    def __init__(self, 
                 model,
                 tokenizer,
                 model_type,
                 model_name,
                 template_conditioned,
                template_document_keyword):
        
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model_type = model_type
        
        prompt_prefix_template = template_conditioned['prompt_prefix_template']
        prompt_suffix_template = template_conditioned['prompt_suffix_template']
        instruction = template_conditioned['instruction']
        
        self.prompt_prefix_ids = self.tokenizer(prompt_prefix_template, return_tensors="pt").input_ids
        self.prompt_suffix_ids = self.tokenizer(prompt_suffix_template, return_tensors="pt").input_ids
        
        #### Strip start and bos token --> we are going to concat ids later anyway 
        self.prompt_prefix_ids = strip_eos_token(self.prompt_prefix_ids)
        
        self.prompt_suffix_ids = strip_bos_token(self.prompt_suffix_ids)
        self.prompt_suffix_ids = strip_eos_token(self.prompt_suffix_ids)
        self.prompt_template_document_keyword = template_document_keyword
            
        print(self.prompt_suffix_ids)
        print('PROMPT PREFIX:', self.tokenizer.decode(self.prompt_prefix_ids[0]))
        print('PROMPT SUFFIX:', self.tokenizer.decode(self.prompt_suffix_ids[0]))

        self.coco = Coco()
        
    def merge_prompt_summary_prefix(self,
                                   prompt_source_ids,
                                   summ_ids):
        #### unconditioned generations
        if prompt_source_ids.shape[-1] == 0:
            if summ_ids.shape[-1] == 0:
                if self.model_name == 'falcon7b':
                    #### Falcon doesn't have a bos token so just use a space as start TODO: bettter strategy
                    input_ids = torch.tensor([tokenizer(' ').input_ids[0]]).unsqueeze(0)
                else:
                    input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)
                
            else:
                input_ids = summ_ids
        else:
            if summ_ids.shape[-1] != 0:
                input_ids = torch.cat([
                    prompt_source_ids.cpu(),
                    summ_ids.cpu()], 
                    dim = -1)
                
            else:
                input_ids = prompt_source_ids
        input_ids = input_ids.to(device)
        return input_ids
        
    def run_model_inference(self,
                           input_ids,
                           max_len):
        attention_mask = torch.tensor([1] * input_ids.shape[-1])
        attention_mask = attention_mask.unsqueeze(0)
#         print(input_ids.shape)
#         print(attention_mask.shape)
        outputs = self.model.generate(input_ids.to('cuda'),
                                      attention_mask = attention_mask.to('cuda'),
                                     max_length = max_len,
                                     return_dict_in_generate = True,
                                     output_scores = True)
        sequences = outputs['sequences']
        prob_summ_id = outputs['scores'][0].float()
        prob_summ_id = torch.softmax(prob_summ_id, dim=-1).cpu().type(torch.float64).numpy()
        return outputs, sequences, prob_summ_id
    
    def generate_token(self,
                      prompt_source_ids,
                      summ_ids,
                      print_prompt = False):
        with torch.no_grad():
            input_ids = self.merge_prompt_summary_prefix(prompt_source_ids = prompt_source_ids,
                                                summ_ids = summ_ids)
            
            
            max_len = input_ids.shape[-1] + 10
            start_idx = input_ids.shape[1]
            outputs, sequences, prob_summ_id = self.run_model_inference(input_ids = input_ids,
                                                                        max_len = max_len)
            candidates = sequences[0][start_idx:]
            gen_summ_id = candidates[0]
            
            if print_prompt:
                print('PROMPT', self.tokenizer.decode(input_ids[0]))
#                 print('DEC INP', decoder_inp)
               
                print('GENERATED', self.tokenizer.decode([gen_summ_id]))
                print('GENERATED TOKENS', candidates)
                
                print('GEN SUMM ID', gen_summ_id, prob_summ_id, max(prob_summ_id[0].tolist()))
                # print('***' * 13)
            return gen_summ_id, prob_summ_id
            

    def get_score_coco(self,
                       document,
                       summary_ids,
                       target_token_id,
                       prob_summ_conditional,
                       keywords):
        prob_summ_conditional = prob_summ_conditional.squeeze(0)

        # print(summary_ids, target_token_id)
        # print(torch.cat([summary_ids, target_token_id.uns]))

        all_summ_ids = target_token_id.unsqueeze(0)
        if summary_ids.shape[-1] != 0:
            all_summ_ids = torch.cat([summary_ids.squeeze(0), target_token_id.unsqueeze(0)])
        summary = self.tokenizer.decode(all_summ_ids)
        
        target_token = summary.split(' ')[-1]
        keywords_i = []
        if target_token and len(target_token) > 1:
            keywords_i = list(set([each for each in keywords if each.startswith(target_token)]))

        # print('TARGET TOKEN', target_token, keywords_i)
        coco_scores = []
        coco_scores_distance = []

        #### Falcon does not have an unknown token
        mask_token = self.tokenizer.unk_token if self.tokenizer.unk_token else ' '
        for mask_strategy in ['token', 'span', 'sent']:
            masked_document = self.coco.mask_document(source_doc = document, 
                                                      masked_token_list = keywords_i, 
                                                      mask_token = mask_token,
                                                      mask_strategy = mask_strategy)
            # if masked_document != document:
            #     print('ALL SUM IDS', all_summ_ids)
            #     print('COCO', target_token, keywords_i)
            #     print('MASKED DOC', '/', mask_strategy, masked_document)
            
            masked_document_ids = self.tokenizer(masked_document, return_tensors="pt").input_ids
            if masked_document_ids[0][0] in [0,1,2]:
                masked_document_ids = masked_document_ids[:, 1:]
            
            masked_document_ids = torch.cat([self.prompt_prefix_ids, 
                                               masked_document_ids,  
                                               self.prompt_suffix_ids], dim = -1)
            _, masked_document_summ_prob = self.generate_token(masked_document_ids,
                                                              summary_ids,
                                                              print_prompt = False)
            masked_document_summ_prob = masked_document_summ_prob.squeeze(0)
            coco_score_diff = prob_summ_conditional[target_token_id] - masked_document_summ_prob[target_token_id]

            # print('COCO DIFF', prob_summ_conditional[target_token_id], masked_document_summ_prob[target_token_id])
            # coco_score_diff = 1 -coco_score_diff 
            coco_score_distance = get_l1(prob_summ_conditional, masked_document_summ_prob)

            coco_scores += [1 - coco_score_diff]
            coco_scores_distance += [coco_score_distance]
            
        return coco_scores, coco_scores_distance
            
    def postprocess_token(self, t_summ_id, pre_t_summ_ids):
        
        target_token = self.tokenizer.decode([t_summ_id])
        prefix_str = ''
        if pre_t_summ_ids.shape[-1] != 0: 
            # print([pre_t_summ_ids ,t_summ_id])
            prefix_str = self.tokenizer.decode([pre_t_summ_ids[-1] ,t_summ_id])
                
        prev_words = prefix_str.split(' ')
        # print(prev_words, target_token)
        if len(prev_words) > 1:
            target_token = ' '+ target_token
        return target_token
    
    def get_score_poel(self,
                      prob_summ_conditional, 
                      prob_summ_prior, 
                      target_token_id):
        
        prob_summ_conditional = prob_summ_conditional.squeeze(0) + 0.01 ### smoothing
        prob_summ_prior = prob_summ_prior.squeeze(0) + 0.01 #### smoothing

        entropy_conditional = entropy(prob_summ_conditional)
        # taken from paper for BART
        # TODO play around with these scores and optimize per model 
        threshold = 3.597
        lambda_v = 0
        if entropy_conditional >= threshold: 
            lambda_v = 0.656

        pmi_score = np.log(prob_summ_conditional[target_token_id]) - np.log(prob_summ_prior[target_token_id])
        poel_score = np.log(prob_summ_conditional[target_token_id]) - (lambda_v  * np.log(prob_summ_prior[target_token_id]))
        pmi_distance = get_l1(prob_summ_conditional, prob_summ_prior)
        return 1 - pmi_score, pmi_distance, 1 - poel_score

    def get_score_harim(self, p_s2s, p_lm, target_token_id):
        lambda_value = 7
        
        p_s2s = p_s2s.squeeze(0)[target_token_id] + 0.01
        p_lm = p_lm.squeeze(0)[target_token_id] + 0.01
        delta = p_s2s - p_lm
        harim = (1 - p_s2s) * (1 - (p_s2s - p_lm))
        # margin_linear = (1-delta) / 2
        # harim = -(1-p_s2s) * margin_linear + 1
        harim_plus = (np.log(p_s2s)) - (lambda_value * harim)
        
        return harim, harim_plus

    def get_prompt_ids(self, doc, summary):
        print('doc check', doc)
        document_keywords = get_document_keywords(doc, num_keywords = 10)
        document_keywords = ', '.join(document_keywords)

        summary = ' '.join([w for w in summary.split(' ') if w.strip()])
        summary_keywords = self.coco.get_masked_token_list(summary)
        
        source_ids = self.tokenizer(doc, return_tensors="pt").input_ids
        source_ids = strip_bos_token(source_ids)
        source_ids = strip_eos_token(source_ids)

        summary_ids = self.tokenizer(summary, return_tensors="pt").input_ids
        summary_ids = strip_bos_token(summary_ids)
        summary_ids = strip_eos_token(summary_ids)
        # summary_ids = summary_ids.squeeze(0)

        prompt_source_ids = torch.cat([self.prompt_prefix_ids,
                               source_ids.cpu(),
                               self.prompt_suffix_ids],
                              dim = -1)
        prompt_source_keyword = f"{self.prompt_template_document_keyword['prompt_template']}".format(keywords = document_keywords)
        # print('PROMPT topic prior', prompt_source_keyword)
        prompt_source_keywords_ids = self.tokenizer(prompt_source_keyword, return_tensors="pt").input_ids

        print('PROMPT', self.tokenizer.decode(prompt_source_ids[0]))
        return_dict = {'document_keywords': document_keywords, 
         'summary_keywords': summary_keywords,
         'source_ids': source_ids,
         'summary_ids': summary_ids,
         'prompt_source_ids': prompt_source_ids,
         'prompt_source_keywords_ids': prompt_source_keywords_ids}
        
        return return_dict

    def get_tokens_labels(self,
                         inconsistent_spans, 
                         summary):
        
        labels = [0] * len(summary.split(' '))
        
        for nonfactual_span in inconsistent_spans:
            processed_nonfactual_span = longest_common_substring_ignore_punctuations(nonfactual_span, summary)
        
            if processed_nonfactual_span != None and re.search(processed_nonfactual_span, summary):
                start_idx, end_idx = re.search(processed_nonfactual_span, summary).span()
                curr_char_idx = 0
            
                for widx, w in enumerate(summary.split(' ')):
                    end_char_idx = curr_char_idx + (len(w) - 1 )
                    assert (summary[curr_char_idx: end_char_idx + 1] == w)
                    label = 0
                    if curr_char_idx>= start_idx and end_char_idx <= end_idx:
                        label = 1
                        labels[widx] = 1
                    curr_char_idx = end_char_idx + 2
            else:
                print('ERR', nonfactual_span, '|', summary)
                print('***'* 13)
                    
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
        
    def get_scores(self, doc, summary, inconsistent_spans):
        
        source_summary_info = self.get_prompt_ids(doc, summary)
        # summary_ids = source_summary_info['summary_ids']
        # summary_ids = summary_ids.squeeze(0)
        summary_ids, summ_tokens_labels = self.get_tokens_labels(inconsistent_spans, summary)
        summary_ids = torch.tensor(summary_ids)
        # print(self.tokenizer.decode(summary_ids))
        # print(summary_ids)
        # print(source_summary_info['summary_ids'].squeeze(0))
        df_dict_token = {

            'target_token_id': [],
            'factual_label': [],
            'target_token_str': [],
            'P_y|x': [],
            'P_y': [],
            'P_ytopic': [],
            'score_pmi': [],
            'score_pmi_topic': [],
            'distance_pmi': [],
            'distance_pmi_topic': [],
            'score_poel': [],
            'score_poel_topic': [],
            'score_harim': [],
            'score_harim_topic': [],
            'score_harim_plus': [],
            'score_harim_plus_topic': [],
            'score_coco_token': [],
            'score_coco_span': [],
            'score_coco_sent': []
        }
        
        for t_idx, t_summ_id in enumerate(summary_ids):
            # print(t_summ_id, summ_tokens_labels[t_idx])
            pre_t_summ_ids = torch.tensor([])
            if t_idx > 0:
                pre_t_summ_ids = summary_ids[:t_idx].unsqueeze(0)

            
            if self.model_name != 'falcon7b':
                target_token = self.postprocess_token(t_summ_id, pre_t_summ_ids.squeeze(0))
            # print(target_token.split(' '))
            else:
                target_token = self.tokenizer.decode(t_summ_id)
                # if len(target_token.split()) > 1:
                #     print(target_token)
                #     target_token = ' '+ target_token.strip()
            
            gen_id, prob_summ_conditional = self.generate_token(prompt_source_ids = source_summary_info['prompt_source_ids'],
                                    summ_ids = pre_t_summ_ids,
                                    print_prompt = False)
            
            _, prob_summ_prior = self.generate_token(prompt_source_ids = torch.tensor([[]]), 
                                                     summ_ids = pre_t_summ_ids,
                                                     print_prompt = False)
            
            
            _, prob_summ_prior_topic = self.generate_token(prompt_source_ids = source_summary_info['prompt_source_keywords_ids'], 
                                                     summ_ids = pre_t_summ_ids,
                                                     print_prompt = False)
            
            
            
            pmi_score, pmi_distance, poel_score = self.get_score_poel(prob_summ_conditional, 
                                                                 prob_summ_prior, 
                                                                 target_token_id = t_summ_id)
            # print('PMI', pmi_score, prob_summ_prior.squeeze(0)[t_summ_id])
            
            pmi_score_topic, pmi_distance_topic, poel_score_topic = self.get_score_poel(prob_summ_conditional, 
                                                                 prob_summ_prior_topic, 
                                                                 target_token_id = t_summ_id)

            # print('PMI TOPIC', pmi_score_topic)
                
            coco_scores, coco_scores_distance = self.get_score_coco(document = doc,
                               summary_ids = pre_t_summ_ids,
                               target_token_id = t_summ_id,
                               prob_summ_conditional = prob_summ_conditional,
                               keywords = source_summary_info['summary_keywords'])
            # print('COCO', coco_scores)

            harim_score, harim_plus_score = self.get_score_harim(p_s2s = prob_summ_conditional, 
                                               p_lm = prob_summ_prior, 
                                               target_token_id = t_summ_id)
            # print('HARIM', harim_score)

            harim_score_topic, harim_plus_score_topic = self.get_score_harim(p_s2s = prob_summ_conditional, 
                                               p_lm = prob_summ_prior_topic, 
                                               target_token_id = t_summ_id)
            
            # print('HARIM', harim_score_topic)

            # print(str(prob_summ_conditional.squeeze(0).tolist()))
            df_dict_token['target_token_id'].append(t_summ_id.item())
            df_dict_token['target_token_str'].append(target_token)
            df_dict_token['factual_label'].append(summ_tokens_labels[t_idx])
            df_dict_token['P_y|x'].append(str(prob_summ_conditional.squeeze(0).tolist()))
            df_dict_token['P_y'].append(str(prob_summ_prior.squeeze(0).tolist()))
            df_dict_token['P_ytopic'].append(str(prob_summ_prior_topic.squeeze(0).tolist()))
            df_dict_token['score_pmi'].append(pmi_score)
            df_dict_token['score_pmi_topic'].append(pmi_score_topic)
            df_dict_token['distance_pmi'].append(pmi_distance)
            df_dict_token['distance_pmi_topic'].append(pmi_distance_topic)
            df_dict_token['score_poel'].append(poel_score)
            df_dict_token['score_poel_topic'].append(poel_score_topic)
            df_dict_token['score_harim'].append(harim_score)
            df_dict_token['score_harim_topic'].append(harim_score_topic)
            df_dict_token['score_harim_plus'].append(harim_plus_score)
            df_dict_token['score_harim_plus_topic'].append(harim_plus_score_topic)
            df_dict_token['score_coco_token'].append(coco_scores[0])
            df_dict_token['score_coco_span'].append(coco_scores[1])
            df_dict_token['score_coco_sent'].append(coco_scores[2])
            # print('***'* 13)
            
            
        return pd.DataFrame(df_dict_token)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-data_folder", 
                           "--data_folder",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets')

    argParser.add_argument("-data_file", 
                           "--data_file",
                          default = 'Genaudit_annotations.csv')

    argParser.add_argument("-template_folder", 
                           "--template_folder",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/prompt_templates')
    
    argParser.add_argument("-template_document_context", 
                           "--template_document_context",
                          default = 'document_context.json')

    argParser.add_argument("-template_context_topic", 
                           "--template_context_topic",
                          default = 'document_topic_keywords.json')

    argParser.add_argument("-model_name", 
                           "--model_name",
                          default = 'mistral7b')
    
    argParser.add_argument("-origin", 
                           "--origin",
                          default = 'XSUM')
    
    args = argParser.parse_args()

    tokenizer, model = load_model(args.model_name)
    
    df_data = pd.read_csv(f'{args.data_folder}/{args.data_file}')
    df_split = df_data[df_data['model'] == args.model_name]
    df_split = df_split[df_split['origin'] == args.origin]

    
    with open(f'{args.template_folder}/{args.template_document_context}', 'r') as fp:
        template_document_context = json.load(fp)
    
    with open(f'{args.template_folder}/{args.template_context_topic}', 'r') as fp:
        template_document_keyword = json.load(fp)

    benchmark_pipeline = Benchmarks(model = model,
           tokenizer = tokenizer,
          model_type = 'decoder',
          model_name = args.model_name,
          template_conditioned = template_document_context,
         template_document_keyword = template_document_keyword )

    df_split_scores = []
    row_ids = []
    for idx, row in tqdm(df_split.iterrows(), total=df_split.shape[0]):
        doc = row['source']
        summary = row['summary']
        inconsistent_spans = row['annotated_spans']
        inconsistent_spans = inconsistent_spans.split('<sep>') if type(inconsistent_spans) is str else []
        df_scores_row = benchmark_pipeline.get_scores(doc, summary, inconsistent_spans)
        df_split_scores.append(df_scores_row)
        row_ids += [row['id']] * len(df_scores_row)
        
    df_split_scores = pd.concat(df_split_scores)
    df_split_scores['document_id'] = row_ids
    print(df_split_scores.head())

    df_split_scores.to_csv(f'/scratch/ramprasad.sa/probing_summarization_factuality/metric_scores/Genaudit/scores_{args.model_name}_{args.origin}.csv')
