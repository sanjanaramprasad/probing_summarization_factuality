from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.utils import get_chatgpt_response, extract_nonfact_spans, load_model
from tqdm import tqdm
import argparse
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()







def generate_summary(document, 
                     model,
                    tokenizer):
    instruction = "Generate a summary for the following document in brief."
    summary_prompt_template = f'{{instruction}}\nDocument: {{doc}}\nSummary: '
    summary_prompt = summary_prompt_template.format(instruction = instruction,
                             doc = document)
    summary_prompt_ids = tokenizer(summary_prompt, return_tensors="pt") 
    summary_prompt_ids = summary_prompt_ids.to('cuda')
    
    outputs = model.generate(**summary_prompt_ids,
                        max_length = summary_prompt_ids.input_ids.shape[-1] +500)
    summary = tokenizer.decode(outputs[0][summary_prompt_ids.input_ids.shape[-1]:])
    summary = summary.strip()
    return summary




def evaluate_inconsistent_sentences(document,
                                   summary):
    instruction = f'Identify and list spans in the summary which are not supported by evidence from the content.\nIf there are no unsupported spans respond with "None"'
    eval_prompt_template = f'{{instruction}}\nContent: {{doc}}\nSummary:{{summ}}\nAnswer:'
    eval_prompt = eval_prompt_template.format(instruction = instruction,
                                         doc = document,
                                         summ = summary)
    
    response = get_chatgpt_response(eval_prompt)
    inconsistent_sentences = extract_nonfact_spans(response)
    return inconsistent_sentences
    
    
def evaluate_inconsistent_spans(document,
                                inconsistent_sentences):
    
    minimaspan_instr = 'List all the minimal inconsistent spans in the following sentence. A minimal inconsistent span is the smallest text span that needs to be modified to make the sentence consistent with the content'
    minimspan_prompt_template = f'Content: {{doc}}\n{{instr}}\nSentence:{{sent}}\nAnswer:'

    inconsistent_spans = []
    for sent in inconsistent_sentences:
        minimspan_prompt = minimspan_prompt_template.format(instr = minimaspan_instr,
                                                    doc = document,
                                                    sent = sent)
        # print(minimspan_prompt)
        minimal_spans = get_chatgpt_response(minimspan_prompt)
        
        minimal_spans = extract_nonfact_spans(minimal_spans)
        inconsistent_spans += minimal_spans
    return inconsistent_spans


def make_tokens_labels(inconsistent_spans,
                      summary,
                     tokenizer):
    
    labels = [0] * len(summary.split(' '))
    for nonfactual_span in inconsistent_spans:
            print(nonfactual_span, summary)
            start_idx, end_idx = re.search(nonfactual_span, summary).span()
            curr_char_idx = 0
            for widx, w in enumerate(summary.split(' ')):
                end_char_idx = curr_char_idx + (len(w) - 1 )
                assert (summary[curr_char_idx: end_char_idx + 1] == w)
                    # print(summary[curr_char_idx: end_char_idx + 1], w)
                label = 0
                if curr_char_idx>= start_idx and end_char_idx <= end_idx:
                    label = 1
                    labels[widx] = 1
                curr_char_idx = end_char_idx + 2
    words_labels = list(zip(summary.split(' '), labels))
    tokens = [1]
    token_labels = [0]
    for w, l in words_labels:
        word_tokens = tokenizer(f'{w}').input_ids[1:]
        tokens += word_tokens
        token_labels += [l] * len(word_tokens)
    return tokens, token_labels
            
def make_xsum_sample():
    dataset = load_dataset("EdinburghNLP/xsum",
                      cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
    df_train_probe = df_train.sample(3500)
    df_train_probe.to_csv('/home/ramprasad.sa/probing_summarization_factuality/datasets/xsum_probe_train.csv')
    return df_train_probe

def make_annotated_data(args):
    if args.make_xsum_sample:
        df_data = make_xsum_sample()
    else:
        df_data = pd.read_csv(args.read_path)
        
    document_key = args.document_key
    
    df_write = {'document' : [],
                'summary': [],
                'nonfactual_spans': []}

    tokenizer, model = load_model(args.model)
    
    for idx, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        summary = generate_summary(document = row[document_key], 
                                   model = model,
                                  tokenizer = tokenizer)
        
        inconsistent_sentences = evaluate_inconsistent_sentences(
                                        document = row[document_key],
                                       summary = summary)
        inconsistent_spans = evaluate_inconsistent_spans(row[document_key],
                                inconsistent_sentences)
        df_write['document'] += [row[document_key]]
        df_write['summary'] += [summary]
        df_write['nonfactual_spans'] += ['<sep>'.join(inconsistent_spans)]
    
    df_write = pd.DataFrame(df_write)
    # write_dir = '/scratch/ramprasad.sa/summarization_probe/probe_datasets/XSUM_GPT'
    write_path = f'{args.write_dir}/{args.write_file}.csv'
    df_write.to_csv(write_path)
    return 


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-read_path", 
                           "--read_path",
                          default = '/home/ramprasad.sa/probing_summarization_factuality/datasets/xsum_probe_train.csv')
    argParser.add_argument("-make_xsum_sample", 
                           "--make_xsum_sample",
                          default = 0)
    
    argParser.add_argument("-model", 
                           "--model",
                          default = 'mistral7b')
    
    argParser.add_argument("-document_key", 
                           "--document_key",
                          default = 'document')
    
    argParser.add_argument("-write_dir", 
                           "--write_dir",
                          default = '/scratch/ramprasad.sa/summarization_probe/probe_datasets')
    
    argParser.add_argument("-write_file", 
                           "--write_file",
                          default = 'XSUM_GPT_dummy')
    args = argParser.parse_args()
    make_annotated_data(args)

