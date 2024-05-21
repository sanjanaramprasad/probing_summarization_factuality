import torch
import os 
import uuid
import pandas as pd
import torch.nn.functional as F
from scripts.utils import load_model
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse

import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def KL_divergence(P,Q):
    # print(P * np.log(P / Q))
    kl_values = P * np.log(P / Q)
    kl_values = kl_values[np.isfinite(kl_values)]  # remove entries where p or q is 0
    kl_div = np.sum(kl_values)
    return kl_div
    
def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return round(0.5 * (KL_divergence(_P, _M) + KL_divergence(_Q, _M)), 5)

''' gets predicted tokens and probabilities of all layers'''
def get_hstates_prediction(hstate, model):
    # print(hstate.shape)
    logits = model.lm_head(hstate.to('cuda'))
    probs = F.softmax(logits, dim=-1).cpu().detach()
    # print(probs.shape)
    predicted_tokens = []
    for layer in probs:
        predicted_tokens.append(torch.tensor([torch.argmax(layer).item()]))
    assert(len(predicted_tokens) == probs.shape[0])
    predicted_tokens = torch.stack(predicted_tokens, dim = 0)
    return predicted_tokens, probs.cpu().detach().numpy()


def make_layer_df(df_dict,
                  layer_num,
                  prob_dist,
                  final_layer_prob,
                  predicted_tokens,
                 target_token):
    
    
    jsd_final_layer = JSD(final_layer_prob, prob_dist)
    
    # print(layer_num, jsd_final_layer)
    prob_dist = prob_dist.tolist()
    
    ### layer predicted token
    pred_prob = max(prob_dist)
    pred_token = predicted_tokens[layer_num].item()
            
    ### final generated token prob
    target_token_prob = prob_dist[target_token]
        
    top3_predicted_probs = sorted(prob_dist, reverse = True)[:3]
    top3_predicted_tokens = [prob_dist.index(each) for each in top3_predicted_probs]
    # if layer_num >  30:
    #     print(layer_num)
    #     print('PRED', pred_prob, pred_token)
    #     print('TGT', target_token, target_token_prob)
    #     print('***' * 13)
    jsd_key = f'jsd_layer{layer_num}'
    df_dict[jsd_key] = [jsd_final_layer]

    pred_prob_layer_key = f'pred_prob_layer{layer_num}'
    df_dict[pred_prob_layer_key] = [pred_prob]
    
    pred_token_layer_key = f'pred_token_layer{layer_num}'
    df_dict[pred_token_layer_key] = [pred_token]

    target_prob_layer_key = f'target_prob_layer{layer_num}'
    df_dict[target_prob_layer_key] = [target_token_prob]

    top3_predicted_probs_key = f'top3_predicted_probs_layer{layer_num}'
    df_dict[top3_predicted_probs_key] = [top3_predicted_probs]

    top3_predicted_tokens_key = f'top3_predicted_tokens_layer{layer_num}'
    df_dict[top3_predicted_tokens_key] = [top3_predicted_tokens]
    return df_dict

def get_token_layer_info_dict(summ_token_hstates,
                              target_token,
                              factual_label,
                              model):
    
    predicted_tokens, probs = get_hstates_prediction(summ_token_hstates, model)
    
    rand_token_id = uuid.uuid4()
    df_dict = {
            'target_token': [],
            'label': []}
    
    df_dict['target_token'].append(f'{rand_token_id}_{target_token}')
    
    for layer_num in range(0, summ_token_hstates.shape[0]):
        if layer_num > 0:
            hidden_state_sim = cosine_similarity(summ_token_hstates[layer_num,:].unsqueeze(0), summ_token_hstates[layer_num- 1,:].unsqueeze(0))
            hidden_state_sim = hidden_state_sim[0][0]
            df_dict[f'prev_hidden_state_sim_layer{layer_num}'] = [hidden_state_sim]
            
    final_layer_prob = probs[-1]
    for layer_num in range(0, predicted_tokens.shape[0]):
            prob_dist = probs[layer_num]
            
            
            df_dict = make_layer_df(df_dict,
                                    layer_num,
                                    prob_dist,
                                    final_layer_prob,
                                    predicted_tokens,
                                    target_token
                                    )
        
    df_dict['label'].append(factual_label)
    return pd.DataFrame(df_dict)
            

def get_summary_layer_info(example, 
                           model):
    # print(example)
    source_len = example['source_len']
    summary_len = example['summary_len']
    hidden_states = example['hidden_states']
    summary_tokens = example['all_tokens'][source_len: source_len + summary_len]
    summary_token_labels = example['summary_token_labels']
    if source_len.item() == 0:
        summary_hidden_states = hidden_states
    else:
        summary_hidden_states = hidden_states[:, source_len - 1: (source_len + summary_len) - 1, :]
    summary_hidden_states = summary_hidden_states.permute(1,0,2) ### we want to iterate of tokens not layers

    df_list = []
    # print(hidden_states.shape, source_len)
    for idx, summ_token_hstates in enumerate(summary_hidden_states):
        # rand_token_id = uuid.uuid4()
        target_token = summary_tokens[idx]
        target_token_label = summary_token_labels[idx]
        
        df_token_layer_info = get_token_layer_info_dict(summ_token_hstates, 
                                                        target_token = target_token,
                                                        factual_label = target_token_label,
                                                        model = model)
        
        df_list.append(df_token_layer_info)
    return pd.concat(df_list)

def make_prob_df(folder_path,
                 model):
    files = os.listdir(folder_path)
    files = sorted(files)
    dfs_data = []
    all_docids = []
    for file_name in tqdm(files):
        docid = file_name.split('#')[-1].split('.pt')[0]
        if os.path.isfile(os.path.join(folder_path, file_name)):
            example = torch.load(os.path.join(folder_path, file_name))
            df_example_dict = get_summary_layer_info(example,
                                                     model)
            all_docids += [docid] * len(df_example_dict)
            dfs_data.append(df_example_dict)
    dfs_data = pd.concat(dfs_data)
    dfs_data['docid'] = all_docids
    return dfs_data



if __name__ == '__main__':
    argParser = argparse.ArgumentParser()



    argParser.add_argument("-model", 
                           "--model",
                          default = 'mistral7b')
    
    argParser.add_argument("-origin", 
                           "--origin",
                          default = 'XSUM')
    
    argParser.add_argument("-prompt_type", 
                           "--prompt_type",
                          default = 'document_context')
    args = argParser.parse_args()
    
    folder_path = f'/scratch/ramprasad.sa/probing_summarization_factuality/internal_states/Genaudit/{args.origin}/{args.model}/{args.prompt_type}'
    
    tokenizer, model = load_model(f'{args.model}')

    df_data = make_prob_df(folder_path=folder_path,
                           model = model)
    
    df_data.to_csv(f'/scratch/ramprasad.sa/probing_summarization_factuality/metric_scores/Genaudit/layer_wise_uncertainty_{args.model}_{args.origin}_document_context.csv')
    print(df_data)