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



def get_nonfactual_spans(before_summary_sents, after_summary_sents):
    before_summary_sents = postprocess(before_summary_sents)
    after_summary_sents = postprocess(after_summary_sents)
    nonfactual_spans_processed = find_edited_spans(before_summary_sents, after_summary_sents)
#     if before_summary_sents != after_summary_sents:
#         print(before_summary_sents)
#         print(after_summary_sents)
#         print(nonfactual_spans_processed)
#         print('**'* 13)
    return nonfactual_spans_processed



def get_summary_sentences(cand_keys, summid):
    source = []
    summary = []
    annotated_spans = []
#     print(cand_keys, len(cand_keys))
    for sent_num in range(0, len(cand_keys)):
        sent_id = f'{summid}:{sent_num}'
        if sent_id in cand_keys:
            
            sent_ann = cand_keys[sent_id]
            if not source:
                source = sent_ann['input_lines']
        
            for summline in sent_ann['prev_summ_lines'] + [sent_ann['before_summary_sent']]:
                        if summline not in summary:
                            summary.append(summline)

            sent_annotated_spans = get_nonfactual_spans(sent_ann['before_summary_sent'],
                                                               sent_ann['after_summary_sent'])
            annotated_spans += sent_annotated_spans
                
    return source, summary, annotated_spans


def make_data_usb(args):
    data = read_jsonl(args.read_path)

    df_dict = {
        'id': [],
        'source': [],
        'summary': [],
        'annotated_spans': [],
        'model': [],
        'origin': []
    }

    for dat in data:
        did = dat['id']
        sentidx = did.split(':')[-1]
        summid = ':'.join(did.split(':')[:-1])
        model = summid.split(':')[-1].split('-')[0]
        origin = summid.split('#')[1].split('-')[0]
        if summid not in df_dict['id']:
            cand_keys = {dat['id'] : dat for dat in data if summid in dat['id']}
            source, summary, annotated_spans = get_summary_sentences(cand_keys, summid)
            
            df_dict['id'].append(summid)
            df_dict['source'].append('. '.join(source))
            df_dict['summary'].append('. '.join(summary))
            df_dict['annotated_spans'].append('<sep>'.join(annotated_spans))
            df_dict['model'].append(model)
            df_dict['origin'].append(origin)
#             break
    df_dict = pd.DataFrame(df_dict)
    df_dict.to_csv(args.write_path)
    return df_dict

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-read_path", "--read_path",      
                           default="/home/ramprasad.sa/probing_summarization_factuality/datasets/annotations/genaudit_data_final_2feb.jsonl")
    
    argParser.add_argument("-write_path", "--write_path",
                           default="/home/ramprasad.sa/probing_summarization_factuality/datasets/Genaudit_annotations.csv")
    args = argParser.parse_args()
    make_data_usb(args)
    



    