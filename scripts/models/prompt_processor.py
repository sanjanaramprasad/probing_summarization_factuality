import json
import torch
import pandas as pd
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from nnsight import LanguageModel, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight.tracing.Proxy import Proxy


model_path = {'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.1',
             'falcon7b': 'tiiuae/falcon-7b-instruct',
             'llama7b': '/work/frink/models/Llama-2-7b-chat-hf',
             'flanul2': 'google/flan-ul2'}


def strip_bos_eos_ids(ids):
    ids = ids[:, 1:] if ids[0][0] in [0,1,2] else ids
    return ids
    

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],
                                         cache_dir = '/scratch/ramprasad.sa/huggingface_models')

    
    return tokenizer

class PromptProcessor:

    def __init__(self,
                 prompt_template,
                 prompt_template_path,
                prompt_type,
                tokenizer):

        self.prompt_template = prompt_template

        with open(f'{prompt_template_path}/{prompt_type}.json', 'r') as fp:
            self.prompt_dict = json.load(fp)
            
        self.instruction = self.prompt_dict['instruction'] if 'instruction' in self.prompt_dict else ''
        self.prefix = self.prompt_dict['prompt_prefix_template'] if 'prompt_prefix_template' in self.prompt_dict else ''
        self.suffix = self.prompt_dict['prompt_suffix_template'] if 'prompt_suffix_template' in self.prompt_dict else ''
        self.tokenizer = tokenizer

    def get_prompt_attributes_idx(self,
                                  prompt_ids,
                                  source,
                                  summary):
        instr_idx = -1
        instr_prefix_idx = -1
        instr_prefix_src_idx = -1
        instr_prefix_src_suffix_idx = -1
        print('attri', len(prompt_ids))
        for span_idx in range( len(prompt_ids)):
            if span_idx %100 == 0:
                print(span_idx)
            span_tokens = prompt_ids[1:span_idx]
            if self.tokenizer.decode(span_tokens) == self.prompt_template.format(instruction = self.instruction,
                                            prompt_prefix = '',
                                            source = '',
                                            prompt_suffix = '',
                                            summary = ''
                                           ).strip():
                instr_idx = span_idx
            
            if self.tokenizer.decode(span_tokens) == self.prompt_template.format(instruction = self.instruction,
                                        prompt_prefix = self.prefix,
                                        source = '',
                                        prompt_suffix = '',
                                        summary = ''
                                       ).strip():
                instr_prefix_idx = span_idx
            
            if self.tokenizer.decode(span_tokens) == self.prompt_template.format(instruction = self.instruction,
                                        prompt_prefix = self.prefix,
                                        source = source,
                                        prompt_suffix = '',
                                        summary = ''
                                       ).strip():
                instr_prefix_src_idx = span_idx
            
            if self.tokenizer.decode(span_tokens) == self.prompt_template.format(instruction = self.instruction,
                                        prompt_prefix = self.prefix,
                                        source = source,
                                        prompt_suffix = self.suffix,
                                        summary = ''
                                       ).strip():
                instr_prefix_src_suffix_idx = span_idx 
                
        return instr_idx, instr_prefix_idx, instr_prefix_src_idx, instr_prefix_src_suffix_idx
    
    def get_nonfactual_span_idx(self,
                                nonfactual_span,
                                summary_tokens,
                                start_idx = -100,
                                end_idx = -100):
        # print(len(summary_tokens), end_idx, nonfactual_span)
        for tok_idx, tok in enumerate(summary_tokens):
            if tok_idx > end_idx:
                tok_str = self.tokenizer.decode(tok)
                if tok_str in nonfactual_span[:len(tok_str)]:
                    start_idx = tok_idx
                
                if self.tokenizer.decode(summary_tokens[start_idx: tok_idx + 1]) == nonfactual_span:
                    end_idx = tok_idx
                    break
        
        return start_idx, end_idx

    def get_summary_labels(self,
                          summary_tokens,
                          nonfactual_spans):
        start_idx = -100
        end_idx = -100
        summary_labels = [0] * len(summary_tokens)
        nonfactual_spans = nonfactual_spans.split('<sep>')

        for i in range(len(nonfactual_spans)):
            nonfactual_span = nonfactual_spans.pop(0)
            start_idx, end_idx = self.get_nonfactual_span_idx(nonfactual_span,
                                             summary_tokens,
                                             start_idx = start_idx,
                                             end_idx = end_idx)
    
            for idx in range(start_idx, end_idx + 1):
                summary_labels[idx] = 1
        return summary_labels
        
    
    def make_prompt_token_labels(self,
                                 source,
                                 summary,
                                 nonfactual_spans):
        # source = ' '.join(source.split(' ')[:50])
        prompt = self.prompt_template.format(instruction = self.instruction,
                                    prompt_prefix = self.prefix,
                                    source = source,
                                    prompt_suffix = self.suffix,
                                    summary = summary
                                   )

        
        prompt_tokens = self.tokenizer(prompt).input_ids
        instr_idx, instr_prefix_idx, instr_prefix_src_idx, instr_prefix_src_suffix_idx = self.get_prompt_attributes_idx(prompt_ids = prompt_tokens,
                                      source = source,
                                      summary = summary)
        
        summary_tokens = prompt_tokens[instr_prefix_src_suffix_idx:]
        
        summary_labels = self.get_summary_labels(summary_tokens,
                          nonfactual_spans)
        print('SUMMARY', summary)
        return_dict = {'prompt': prompt,
                       'instruction': self.instruction,
                       'prompt_prefix': self.prefix,
                       'source': source,
                       'summary': summary,
                       'prompt_template': self.prompt_template,
                       'prompt_suffix': self.suffix,
                      'instr_idx': instr_idx,
                      'instr_prefix_idx': instr_prefix_idx,
                      'instr_prefix_src_idx': instr_prefix_src_idx,
                      'instr_prefix_src_suffix_idx': instr_prefix_src_suffix_idx,
                      'summary_labels': summary_labels}
        return return_dict


    
if __name__ == '__main__':
        prompt_template = f'{{instruction}}{{prompt_prefix}}{{source}}{{prompt_suffix}}{{summary}}'

        prompt_template_path = '/home/ramprasad.sa/probing_summarization_factuality/datasets/prompt_templates/'
        prompt_type = 'document_context_causal'

        with open(f'{prompt_template_path}/{prompt_type}.json', 'r') as fp:
            prompt_dict = json.load(fp)
    
        tokenizer = load_tokenizer('mistral7b')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model_name = 'mistral7b'
        genaudit_read_path = '/home/ramprasad.sa/probing_summarization_factuality/datasets/Genaudit_annotations.csv'
        df_genaudit = pd.read_csv(genaudit_read_path)
        df_genaudit_mistral = df_genaudit[df_genaudit['model'] == model_name]

        prompt_processor = PromptProcessor(prompt_template = prompt_template,
                                   prompt_template_path = prompt_template_path,
                                   prompt_type = prompt_type,
                                   tokenizer = tokenizer
                                  )
        
        for idx, row in df_genaudit_mistral[~df_genaudit_mistral['annotated_spans'].isnull()].iterrows():
            source = row['source']
            summary = row['summary']
            nonfactual_spans = row['annotated_spans']
            prompt_dict = prompt_processor.make_prompt_token_labels(source= source,
                                                                    summary = summary,
                                                                    nonfactual_spans = nonfactual_spans)
            print(prompt_dict)
            break