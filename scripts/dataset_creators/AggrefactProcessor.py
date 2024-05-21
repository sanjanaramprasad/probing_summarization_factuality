import pandas as pd 
from scripts.utils import read_jsonl, generate_random_id
from datasets import load_dataset

def split_into_contiguous_sublists(lst):
    if not lst:
        return []

    # Sort the list to handle unordered input
    lst.sort()

    sublists = []
    current_sublist = [lst[0]]
    # print(lst)
    for i in range(1, len(lst)):
        # print(lst[i])
        if lst[i] == lst[i - 1] + 1:
            current_sublist.append(lst[i])
        else:
            sublists.append(current_sublist)
            current_sublist = [lst[i]]

    sublists.append(current_sublist)
    
    return sublists


class AggrefactProcessor:

    def __init__(self):
        self.xsum_dataset = load_dataset("EdinburghNLP/xsum", cache_dir = '/scratch/ramprasad.sa/huggingface_datasets')
        return 
    

    def __get_xsum_error_annotated_category_maps(self,
                                                 shortlisted_spans,
                                                 df_uid_model):
        
        span_category_maps = {}
        for _, row in df_uid_model.iterrows():
            hallu_span = row['hallucinated_span'] 
            if type(hallu_span) is str:
                if hallu_span not in span_category_maps:
                    span_category_maps[hallu_span ] = []
                span_category_maps[hallu_span] += [row['hallucination_type']]
        
        shortlisted_span_categories = []
        # print(shortlisted_spans, span_category_maps)
        for shortlisted_span in shortlisted_spans:
            possible_categories = []
            for span, categ in span_category_maps.items():
                if shortlisted_span in span or span in shortlisted_span:
                    possible_categories += categ
            error_cat = 'intrinsic' if possible_categories.count('intrinsic') > possible_categories.count('extrinsic') else 'extrinsic'
            shortlisted_span_categories += [error_cat]
        return shortlisted_span_categories
     
    def __get_xsumfaith_agreed_spans(self,
                                     df_uid_model):
        num_workers = list(set(df_uid_model['worker_id']))
        summary = df_uid_model['summary'].values[0]

        annotated_spans = []
        for worker_idx in range(len(num_workers)):
            df_uid_model_worker = df_uid_model[df_uid_model['worker_id'] == f'wid_{worker_idx}']

            for _, worker_annotations in df_uid_model_worker.iterrows():
                start_span = worker_annotations['hallucinated_span_start']
                end_span = worker_annotations['hallucinated_span_end']
                span_idx = [i for i in range(start_span, end_span)] if start_span > -1 else []
                annotated_spans += span_idx

        shortlisted_spans = [span_idx for span_idx in list(set(annotated_spans)) \
                                if annotated_spans.count(span_idx) == len(num_workers)]
        shortlisted_spans = split_into_contiguous_sublists(shortlisted_spans)
        shortlisted_spans = [summary[span[0]-1 :span[-1]].strip() for span in shortlisted_spans]

        shortlisted_span_categories = self.__get_xsum_error_annotated_category_maps(shortlisted_spans,
                                                                                    df_uid_model)

        return shortlisted_spans, shortlisted_span_categories
    

    def __get_cliff_nonfactual_spans_categories(self,
                                                summary_words,
                                                summary_labels):
        nonfactual_spans = []
        nonfactual_spans_categories = []
            
        span = []
        category = []
        for wid, word in enumerate(summary_words):
            word_label = summary_labels[wid]
            if word_label != 'correct':
                if wid == 0 or summary_labels[wid - 1] == 'correct' or summary_labels[wid -1] != word_label:
                    if span:
                        nonfactual_spans.append(' '.join(span))
                        nonfactual_spans_categories.append(category[-1])
                    span = [word]
                    category = [word_label]
                else:
                    span += [word]
                    category += [word_label]
        return nonfactual_spans, nonfactual_spans_categories


    def process_cliff(self,
                      cliff_read_path,
                      cliff_write_path
                      ):
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
            'summary': []
        }

        cliff_data = read_jsonl(cliff_read_path)
        source_format = 'document'
        benchmark_dataset_name = 'CLIFF'
        origin = 'XSUM'
        summary_origin = 'XFORMER'
        error_origin = 'natural'
        error_type = 'spans'
        error_categorization = 'fine_grained'
        model = 'BART'
        
        for dat in cliff_data:
            dat_id = generate_random_id()
            origin = dat['media_source']
            source_document = dat['document']

            summary_words = dat['bart']['words']
            summary_labels = dat['bart']['label']

            summary = ' '.join(summary_words)
            
            nonfactual_spans, nonfactual_spans_categories = self.__get_cliff_nonfactual_spans_categories(summary_words = summary_words,
                                                    summary_labels = summary_labels)
            
            df_dict['id'].append(dat_id)
            df_dict['benchmark_dataset_name'].append(benchmark_dataset_name)
            df_dict['origin'].append(origin)
            df_dict['source_format'].append(source_format)
            df_dict['summary_origin'].append(summary_origin)
            df_dict['error_origin'].append(error_origin)
            df_dict['error_type'].append(error_type)
            df_dict['error_categorization'].append(error_categorization)
            df_dict['model'].append(model)
            df_dict['nonfactual_spans'].append('<sep>'.join(nonfactual_spans))
            df_dict['nonfactual_spans_category'].append(nonfactual_spans_categories)
            df_dict['source'].append(source_document)
            df_dict['summary'].append(summary)

        df_dict = pd.DataFrame(df_dict)
        df_dict.to_csv(cliff_write_path)
        return df_dict  
        

    
    def process_xsumfaith(self,
                          xsum_read_path,
                          xsum_write_path):
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
            'summary': []
        }

        
        df_xsum_docs = pd.concat([pd.DataFrame( self.xsum_dataset['train']),
                              pd.DataFrame( self.xsum_dataset['validation']), 
                              pd.DataFrame( self.xsum_dataset['test'])])

        df_xsumfaith = pd.read_csv(xsum_read_path)

        benchmark_dataset_name = 'XSUMFaith'
        origin = 'XSUM'
        source_format = 'document'
        summary_origin = 'XFORMER'
        error_origin = 'natural'
        error_type = 'spans'
        error_categorization = 'fine_grained'

        unique_ids = list(set(df_xsumfaith['bbcid']))

        #### each unique id has multiple model summaries 
        for uid in unique_ids:
            source_doc = df_xsum_docs[df_xsum_docs['id'] == str(uid)]
            assert(len(source_doc) != 0)
            df_xsum_uid = df_xsumfaith[df_xsumfaith['bbcid'] == uid]
            uid_models  = list(set(df_xsum_uid['system'].values))
            #### each unique model summary
            for uid_model in uid_models:
                df_uid_model = df_xsum_uid[df_xsum_uid['system'] == uid_model]
                summary = df_uid_model['summary'].values[0]
                summary_origin = 'XFORMER' if uid_model in ['BERTS2S', 'TConvS2S', 'TranS2S', 'PtGen' ] else 'Reference'
                nonfactual_spans, nonfactual_spans_categories = self.__get_xsumfaith_agreed_spans(df_uid_model)

                df_dict['id'].append(uid)
                df_dict['benchmark_dataset_name'].append(benchmark_dataset_name)
                df_dict['origin'].append(origin)
                df_dict['source_format'].append(source_format)
                df_dict['summary_origin'].append(summary_origin)
                df_dict['error_origin'].append(error_origin)
                df_dict['error_type'].append(error_type)
                df_dict['error_categorization'].append(error_categorization)
                df_dict['model'].append(uid_model)
                df_dict['nonfactual_spans'].append('<sep>'.join(nonfactual_spans))
                df_dict['nonfactual_spans_category'].append(nonfactual_spans_categories)
                df_dict['source'].append(source_doc)
                df_dict['summary'].append(summary)

        df_dict = pd.DataFrame(df_dict)
        df_dict.to_csv(xsum_write_path)
        return df_dict 
