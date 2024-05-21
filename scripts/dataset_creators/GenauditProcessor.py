import json
import string
import difflib
import pandas as pd
from tqdm import tqdm
from scripts.utils import read_jsonl
import argparse
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
# Get the English stopwords list from NLTK
stop_words = set(stopwords.words('english'))

def postprocess(txt):
    txt = [w.strip(string.punctuation) for w in txt.split(' ')]
    txt = ' '.join(txt)
    return txt

def find_edited_spans(s1, s2):
    
    s1 = s1.strip(string.punctuation)
    s2 = s2.strip(string.punctuation)
    differ = difflib.Differ()
    diff = list(differ.compare(s1.split(), s2.split()))

    edited_spans = []
    current_span = ""
    for item in diff:
        code, word = item[0], item[2:]
        
        if code == ' ':
            if current_span:
                # print(current_span)
                # if not len([w for w in current_span.split() if w not in stop_words]):
                #     print(current_span)
                    # printed = True
                if len([w for w in current_span.split() if w not in stop_words]):
                    edited_spans.append(current_span.strip())
                current_span = ""
        elif code == '-':
            current_span += word + " "

    
    if current_span:
        if len([w for w in current_span.split() if w not in stop_words]):
            edited_spans.append(current_span.strip())
    
    return edited_spans


class GenAuditProcessor:

    def __init__(self):
        self.benchmark_dataset_name = 'Genaudit'
        self.summary_origin = 'LLM'
        self.error_origin = 'natural'
        self.error_type = 'spans'
        self.error_categorization = 'binary'
        return 
    

    def __get_nonfactual_spans(self,
                             before_summary_sents, 
                             after_summary_sents):
        before_summary_sents = postprocess(before_summary_sents)
        after_summary_sents = postprocess(after_summary_sents)
        nonfactual_spans_processed = find_edited_spans(before_summary_sents, after_summary_sents)
    #     if before_summary_sents != after_summary_sents:
    #         print(before_summary_sents)
    #         print(after_summary_sents)
    #         print(nonfactual_spans_processed)
    #         print('**'* 13)
        return nonfactual_spans_processed

    def __get_summary_sentences(self, 
                              cand_keys, 
                              summid):
        source = []
        summary = []
        annotated_spans = []
        # print(cand_keys, len(cand_keys))
        for sent_num in range(0, len(cand_keys)):
            sent_id = f'{summid}:{sent_num}'
            if sent_id in cand_keys:
                sent_ann = cand_keys[sent_id]
                if not source:
                    source = sent_ann['input_lines']
            
                for summline in sent_ann['prev_summ_lines'] + [sent_ann['before_summary_sent']]:
                            if summline not in summary:
                                summary.append(summline)

                sent_annotated_spans = self.__get_nonfactual_spans(sent_ann['before_summary_sent'],
                                                                sent_ann['after_summary_sent'])
                annotated_spans += sent_annotated_spans
                    
        return source, summary, annotated_spans

    def make_data(self,
                  read_path,
                  write_path):
        data = read_jsonl(read_path)
        df_dict = {
            'id': [],
            'benchmark_dataset_name': [],
            'origin': [],
            'source_format': [],
            'summary_origin': [],
            'error_origin': [],
            'error_type': [],
            'error_categorization': [],
            'model': [],
            'nonfactual_spans': [],
            'nonfactual_spans_category': [],
            'source': [],
            'summary': [],

        }

        for dat in data:
            dat_id = dat['id']
            summid = ':'.join(dat_id.split(':')[:-1])
            model = summid.split(':')[-1].split('-')[0]
            origin = summid.split('#')[1].split('-')[0]

            if summid not in df_dict['id']:
                cand_keys = {dat['id'] : dat for dat in data if summid in dat['id']}
                source, summary, annotated_spans = self.__get_summary_sentences(cand_keys, summid)
                source_format = 'document' if origin != 'ACIBENCH' else 'dialogue'
                df_dict['id'].append(dat_id)
                df_dict['benchmark_dataset_name'].append(self.benchmark_dataset_name)
                df_dict['origin'].append(origin)
                df_dict['source_format'].append(source_format)
                df_dict['summary_origin'].append(self.summary_origin)
                df_dict['error_origin'].append(self.error_origin)
                df_dict['error_type'].append(self.error_type)
                df_dict['error_categorization'].append(self.error_categorization)
                df_dict['model'].append(model)
                df_dict['nonfactual_spans'].append('<sep>'.join(annotated_spans))
                df_dict['nonfactual_spans_category'].append([None] * len(annotated_spans))
                df_dict['source'].append(source)
                df_dict['source'].append(source)
        df_dict = pd.DataFrame(df_dict)
        df_dict.to_csv(write_path)
        return df_dict



        