import torch
import json
import argparse
import pandas as pd
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.models.prompt_processor import PromptProcessor
from scripts.utils import load_model
from scripts.dataset_creators.coco_mask import Coco


prompt_template = f'{{instruction}}{{prompt_prefix}}{{source}}{{prompt_suffix}}{{summary}}'


def get_nnsight_model_wrapper(model_name):
    tokenizer, model = load_model(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(['[MASK]'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    model = LanguageModel(model, tokenizer=tokenizer, device_map="auto", dispatch=True)
    return tokenizer, model

def get_clean_logits(prompt,
                    model,
                    N_LAYERS):
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            
            clean_logits = model.lm_head.output[0]
            clean_logits = F.softmax(clean_logits, dim = 1).detach().cpu().save()
            
            clean_hs = [
                model.model.layers[layer_idx].output[0].detach().cpu().save()
                for layer_idx in range(N_LAYERS)
            ]
            
            clean_input_embeddings = model.model.embed_tokens.output.detach().cpu().save()
            
    return clean_input_embeddings, clean_hs, clean_logits



class MakeDataAIE:

    def __init__(self,
                model_name,
                prompt_template,
                prompt_template_path,
                prompt_type
                 ):
        
        self.tokenizer, self.model = get_nnsight_model_wrapper(model_name)

        self.prompt_processor  = PromptProcessor(prompt_template = prompt_template,
                                   prompt_template_path = prompt_template_path,
                                   prompt_type = prompt_type,
                                   tokenizer = self.tokenizer
                                  )
        self.coco = Coco()

    # def get_clean_logits(self,
    #                prompt):
    #     N_LAYERS = len(self.model.model.layers)
        
    #     with self.model.trace() as tracer:
    #         with tracer.invoke(prompt) as invoker:
    #             clean_logits = self.model.lm_head.output[0]
    #             clean_logits = F.softmax(clean_logits, dim = 1).detach().cpu().save()
            
    #             clean_hs = [
    #                 self.model.model.layers[layer_idx].output[0].detach().cpu().save()
    #                 for layer_idx in range(N_LAYERS)
    #             ]
            
    #             clean_input_embeddings = self.model.model.embed_tokens.output.detach().cpu().save()
            
    #     return clean_input_embeddings, clean_hs, clean_logits
    
    # def get_corrupted_logits(self,
    #                         corrupted_prompt):
    #     N_LAYERS = len(self.model.model.layers)
    #     with self.model.trace() as tracer:
    #         with tracer.invoke(corrupted_prompt) as invoker:
    #             corrupted_logits = self.model.lm_head.output[0]
    #             corrupted_logits = F.softmax(corrupted_logits, dim = 1).detach().cpu().save()
            
    #             corrupted_hs = [
    #                 self.model.model.layers[layer_idx].output[0].detach().cpu().save()
    #                 for layer_idx in range(N_LAYERS)
    #             ]
            
    #             noised_embeddings = self.model.model.embed_tokens.output.detach().cpu().save()
    #     return noised_embeddings, corrupted_hs, corrupted_logits

    def get_logits(self,
                   prompt):
        N_LAYERS = len(self.model.model.layers)
        with self.model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                logits = self.model.lm_head.output[0]
                logits = F.softmax(logits, dim = 1).detach().cpu().save()

                hidden_states = [
                    self.model.model.layers[layer_idx].output[0].detach().cpu().save()
                    for layer_idx in range(N_LAYERS)
                ]

                embeddings = self.model.model.embed_tokens.output.detach().cpu().save()
        return embeddings, hidden_states, logits

    def get_patched_logits(self,
                           corrupted_prompt,
                           clean_hidden_states,
                           layer_idx,
                           token_idx):
        
        N_LAYERS = len(self.model.model.layers)
        with self.model.trace() as tracer:
            with tracer.invoke(corrupted_prompt) as invoker:
                self.model.model.layers[layer_idx].output[0].t[token_idx] = clean_hidden_states[layer_idx].t[token_idx]
                patched_logits = self.model.lm_head.output[0]
                patched_logits = F.softmax(patched_logits, dim = 1).detach().cpu().save()
                patched_hs = [
                    self.model.model.layers[layer_idx].output[0].detach().cpu().save()
                    for layer_idx in range(N_LAYERS)
                ]
                
                patched_embeddings = self.model.model.embed_tokens.output.detach().cpu().save()
                
        return patched_embeddings, patched_hs, patched_logits
    
    
    
    # def corrupt_prompt_by_idx(self,
    #                    prompt_tokens,
    #                    corruption_token,
    #                    corruption_idx):
        

    #     corrupted_tokens = [repl_token if tok_idx in corruption_idx else tok.item() for tok_idx, tok in enumerate(prompt_tokens)] 
    #     corrupted_prompt = self.tokenizer.decode(corrupted_tokens)
    #     return corrupted_prompt
            
    ''' '''

    def get_corrupted_prompt(self,
                             tgt_token,
                             clean_tokens,
                             prompt_dict,
                             corruption_strategy):
        
        corruption_token_str = '[MASK]'
        corruption_token = self.tokenizer(corruption_token_str).input_ids[1:]
        assert(len(corruption_token) == 1)
        corruption_token = corruption_token[0]
        corrupted_prompt = prompt_dict['prompt']

        if corruption_strategy == 'instruction':
            corruption_idx = [i for i in range(prompt_dict['instr_idx'])]
            corrupted_prompt = self.corrupt_prompt(prompt_tokens = clean_tokens,
                                               corruption_token = corruption_token,
                                               corruption_idx = corruption_idx)
        



        
        elif corruption_strategy == 'coco_mask':
            # print(clean_tokens)
            clean_tokens = clean_tokens.tolist()
            corrupted_tokens = clean_tokens
            if tgt_token in clean_tokens:
                found_idx = [idx for idx, tok in enumerate(clean_tokens) if (tok == tgt_token) and (idx > prompt_dict['instr_prefix_idx'] and idx < prompt_dict['instr_prefix_src_idx'])]
            
            for fid in found_idx:
                for cid in range(len(clean_tokens)):
                    if cid >= fid - 3 and cid <=fid + 3:
                        corrupted_tokens[cid] = corruption_token
            # print(len(corrupted_tokens), len(clean_tokens))
            # print('CORR TOKENS', corrupted_tokens)
            # print('CLEAN TOKENS', clean_tokens)
            corrupted_prompt = self.tokenizer.decode(corrupted_tokens)
            corrupted_prompt = corrupted_prompt.split('<s>')[-1].strip()
            # summary = prompt_dict['summary']

            # tgt_token_str = self.tokenizer.decode(tgt_token)
            # summary_keywords = self.coco.get_masked_token_list(summary)
            # keywords_i = []
            # # print(summary_keywords)
            # if tgt_token_str and len(tgt_token_str) > 1:
            #     keywords_i = list(set([each for each in summary_keywords if each.startswith(tgt_token_str)]))
            
            #     masked_document = self.coco.mask_document(source_doc = prompt_dict['source'], 
            #                                           masked_token_list = keywords_i, 
            #                                           mask_token = corruption_token_str,
            #                                           mask_strategy ='span')
            
            #     corrupted_prompt = prompt_dict['prompt_template'].format(instruction = prompt_dict['instruction'],
            #                                          prompt_prefix = prompt_dict['prompt_prefix'],
            #                                          source = masked_document,
            #                                          prompt_suffix = prompt_dict['prompt_suffix'],
            #                                         summary = prompt_dict['summary'])
                                                    
        return corrupted_prompt
        
        
    def get_indirect_effect(self,
                            prompt_dict,
                            corruption_strategy = 'coco_mask'):
        prompt = prompt_dict['prompt']
        with self.model.trace() as tracer:
            with tracer.invoke(prompt_dict['prompt']) as invoker:
                clean_tokens = self.model.input[1]["input_ids"].squeeze().save()

        N_LAYERS = len(self.model.model.layers)
        summary_tokens = clean_tokens[prompt_dict['instr_prefix_src_suffix_idx']:]
        summary_labels = prompt_dict['summary_labels']
        assert(self.tokenizer.decode(summary_tokens) == prompt_dict['summary'])
            
        
        clean_input_embeddings, clean_hidden_states, clean_logits = self.get_logits( prompt = prompt)

        summ_it = 0
        summary_token_patching_results = []
        ''' iterate through each summary token as a tgt to patch'''
        for tidx, tgt_token in tqdm(enumerate(clean_tokens), total = len(clean_tokens)):
            '''end of inp prompt'''
            if tidx >= prompt_dict['instr_prefix_src_suffix_idx']:
                # if tidx > 506:
                assert(tgt_token == summary_tokens[summ_it])

                corrupted_prompt = self.get_corrupted_prompt(tgt_token=tgt_token,
                                                                clean_tokens=clean_tokens,
                                                                prompt_dict=prompt_dict,
                                                                corruption_strategy=corruption_strategy)


                # print()
                layer_wise_patching_results = []
                ''' iterate all layers'''
                for layer_idx in range(len(self.model.model.layers)):
                                
                    _, corrupted_hidden_states, corrupted_logits = self.get_logits(prompt=corrupted_prompt)
                    _, patched_hidden_states, patched_logits =  self.get_patched_logits(corrupted_prompt = corrupted_prompt,
                                                                                clean_hidden_states = clean_hidden_states,
                                                                                layer_idx = layer_idx,
                                                                                token_idx = tidx - 1)


                    #### check if while predicting tgt token the clean runs layer is same as pattched run layer
                    assert(  torch.allclose(clean_hidden_states[layer_idx][:, tidx - 1, :], patched_hidden_states[layer_idx][:, tidx - 1, :]))
                    try:
                        assert(corrupted_logits.shape == clean_logits.shape )
                    except Exception as e:
                        print('COMPATIBLE ERROR')
                        print(corrupted_logits.shape, clean_logits.shape)
                        # if corrupted_prompt != prompt:
                        print('CORRUPTED PROMPT', self.tokenizer.decode(tgt_token), corrupted_prompt)
                        with self.model.trace() as tracer:
                            with tracer.invoke(corrupted_prompt) as invoker:
                                corrupted_tokens = self.model.input[1]["input_ids"].squeeze().save()
                        print('CORR-2', corrupted_tokens[:100], clean_tokens[:100]
                              )
                        break

                    
                            

                    append_dict = {
                                    'layer': layer_idx,
                                    'target': tgt_token.item(),
                                    'predicted': torch.argmax(clean_logits[tidx - 1]).item(),
                                    'factual_label': summary_labels[summ_it],
                                    'prob_clean': clean_logits[tidx - 1][tgt_token].item(),
                                    'prob_corrupted': corrupted_logits[tidx - 1][tgt_token].item(),
                                    'prob_patched': patched_logits[tidx - 1][tgt_token].item()
                                    }
                    layer_wise_patching_results.append(append_dict)

                if layer_wise_patching_results:
                    summary_token_patching_results.append(layer_wise_patching_results)
                summ_it += 1

        return summary_token_patching_results
    
    
    # def get_indirect_effect(self,
    #                         prompt_dict,
    #                         corruption_strategy = ''):

    #     with self.model.trace() as tracer:
    #         with tracer.invoke(prompt_dict['prompt']) as invoker:
    #             clean_tokens = self.model.input[1]["input_ids"].squeeze().save()

        
    #     clean_input_embeddings, clean_hs, clean_logits = self.get_clean_logits( prompt = prompt)

        
        
    #     summary_labels = prompt_dict['summary_labels']
    #     summary_tokens = clean_tokens[prompt_dict['instr_prefix_src_suffix_idx']:]
    #     corruption_idx = [i for i in range(prompt_dict['instr_idx'])]
    #     assert(summary_tokens.shape[-1] == len(summary_labels))

    #     corrupted_prompt = self.corrupt_prompt(prompt_tokens = clean_tokens,
    #                                            corruption_idx = corruption_idx)

    #     clean_input_embeddings, clean_hs, clean_logits = self.get_clean_logits( prompt = prompt)

    #     '''noising all tokens of the instructon'''
    #     noised_embeddings, corrupted_hs, corrupted_logits = self.get_corrupted_logits(corrupted_prompt=corrupted_prompt)

    #     summary_token_patching_results = []
    #     summ_idx = 0
    #     for tidx, tgt_token in enumerate(clean_tokens):
    #             ''' after instr, prefix, src, suffix is summary tokens'''
    #             if tidx >= prompt_dict['instr_prefix_src_suffix_idx']:
    #                 assert(tgt_token == summary_tokens[summ_idx])
                    

    #                 layer_wise_patching_results = []
    #                 ''' iterate all layers'''
    #                 for layer_idx in range(len(self.model.model.layers)):
                            
    #                         _, patched_hs, patched_logits =  self.get_patched_logits(corrupted_prompt = corrupted_prompt,
    #                                                                         clean_hs = clean_hs,
    #                                                                         layer_idx = layer_idx,
    #                                                                         token_idx = tidx - 1)
                            
    #                         #### check if while predicting tgt token the corrupted runs layer is not same as pattched run layer
    #                         assert( not torch.allclose(corrupted_hs[layer_idx][:, tidx - 1, :], patched_hs[layer_idx][:, tidx - 1, :]))

    #                         #### check if while predicting tgt token the clean runs layer is same as pattched run layer
    #                         assert(  torch.allclose(clean_hs[layer_idx][:, tidx - 1, :], patched_hs[layer_idx][:, tidx - 1, :]))
                            
                            

    #                         append_dict = {
    #                             'layer': layer_idx,
    #                             'target': tgt_token.item(),
    #                             'predicted': torch.argmax(clean_logits[tidx - 1]).item(),
    #                             'factual_label': summary_labels[summ_idx],
    #                             'prob_clean': clean_logits[tidx - 1][tgt_token].item(),
    #                             'prob_corrupted': corrupted_logits[tidx - 1][tgt_token].item(),
    #                             'prob_patched': patched_logits[tidx - 1][tgt_token].item()
    #                         }
    #                         layer_wise_patching_results.append(append_dict)
                        

    #                 summary_token_patching_results.append(layer_wise_patching_results)
    #                 summ_idx += 1

    #     return summary_token_patching_results

    def write_patched_results(self,
                              summary_patching_results,
                              write_path):
        with open(write_path, 'w') as file:
            for item in summary_patching_results:
                json.dump(item, file)  # Write JSON object
                file.write('\n')
        return

    def get_layerwise_causal_analysis(self,
                                      df,
                                      write_path):
            
            for idx, row in tqdm(df[~df['annotated_spans'].isnull()].iterrows(), total = len(df[~df['annotated_spans'].isnull()])):
                uid = row['id']
                uid = '_'.join(uid.split())
                source = row['source']
                summary = row['summary']
                print(summary)
                nonfactual_spans = row['annotated_spans']
                prompt_dict = self.prompt_processor.make_prompt_token_labels(source= source,
                                                                        summary = summary,
                                                                        nonfactual_spans = nonfactual_spans)
                print(prompt_dict)
                summary_patching_results = self.get_indirect_effect(prompt_dict= prompt_dict)
                filename = f'{write_path}/{uid}.jsonl'
                self.write_patched_results(summary_patching_results = summary_patching_results,
                                           write_path = filename)
                break
            return
                

            

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-origin", 
                           "--origin",
                          default = 'XSUM')
    
    argParser.add_argument("-model_name", 
                           "--model_name",
                          default = 'mistral7b')

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
                          default = 'document_context_causal')
    
    argParser.add_argument("-prompt_template_path", 
                           "--prompt_template_path",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/prompt_templates/')
    
    argParser.add_argument("-read_path", 
                           "--read_path",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/Genaudit_annotations.csv')
    

    args = argParser.parse_args()

    
    # genaudit_read_path = '/home/ramprasad.sa/probing_summarization_factuality/datasets/Genaudit_annotations.csv'

    write_path = f'/scratch/ramprasad.sa/probing_summarization_factuality/causal_analysis/noised_instruction/{args.origin}/{args.model_name}'
    df = pd.read_csv(args.read_path)
    df = df[df['model'] == args.model_name]
    df = df[df['origin'] == args.origin]


    shortlist_ids = ['Cynthia Lamanda#XSUM-19389161:mistral7b-ul2']
    df = df[df['id'].isin(shortlist_ids)]

    nnsight_patcher = MakeDataAIE(model_name = args.model_name,
                prompt_template = prompt_template,
                prompt_template_path = args.prompt_template_path,
                prompt_type = args.prompt_type)


    print(df)
    nnsight_patcher.get_layerwise_causal_analysis(df,
                                                  write_path = write_path)
    