import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from scripts.dataset_creators.read_internal_states import HiddenStatesDataset
from scripts.eval.run_token_scoring import score_predictions_labels
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def compute_scores(labels, predictions):
    auc_score = roc_auc_score(labels, predictions)
    predictions_binary = [0 if each > 0.5 else 1 for each in predictions]
    bacc_score = balanced_accuracy_score(labels, predictions_binary)
    return {'auc': auc_score, 'bacc': bacc_score}



# Define a simple linear probe model
class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionProbe, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)
    
def run_model(dat,
             model):
    hstate, tokens, labels = dat
    nonzero_rows_mask = torch.any(hstate != 0, dim=1)
    hstate_filtered = hstate[nonzero_rows_mask] 
    outputs = model(hstate_filtered.float())
    
    labels = labels[nonzero_rows_mask]
    
    return outputs, labels

def compute_loss(criterion,
                 labels,
                 outputs,
                 class_weights, 
                 ):
    
    label_weights = torch.tensor([class_weights[lab.item()] for lab in labels])
    loss = criterion(outputs.squeeze(), labels.float()) 
    loss = loss * label_weights
    loss = torch.mean(loss)
    
    return loss

def score_dataset(data,
                  model):
    predictions = []
    labels = []
    for dat_idx, dat in enumerate(data):
        out, lab = run_model(dat,
                             model)
        predictions += out.detach().numpy().squeeze().tolist()
        labels += lab.tolist()
    scores_dict = compute_scores(labels, predictions)
    return scores_dict

def run_train(train_data,
              val_data,
              test_data,
              class_weights,
              num_epochs,
              learning_rate,
              write_dir):

    hstate, tok, lab = train_data[0]
    input_size = hstate.size(1)
    output_size = 1  
    
    model = LogisticRegressionProbe(input_size)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    val_score = 0
    for epoch in tqdm(range(0, num_epochs)):
            dat_idx = epoch%len(train_data)
            dat = train_data[dat_idx]


            ########## Train/backprop ########
            out, lab = run_model(dat, 
                            model)
            
            loss = compute_loss(criterion,
                 lab,
                 out,
                 class_weights)
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backward pass
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                val_scores_dict = score_dataset(val_data,
                                                model)
                print('Validation scores', val_scores_dict)
                test_scores_dict = score_dataset(test_data,
                                                model)
                
                print('Test scores', test_scores_dict)
            
            filename = f"test_{test_scores_dict['bacc'].item():.4f}_epoch{epoch}"

            if val_scores_dict['bacc'] > val_score:
                val_score = val_scores_dict['bacc']
                torch.save(model.state_dict(), f'{write_dir}/{filename}')

    return




if __name__ == '__main__':
    argParser = argparse.ArgumentParser()


    argParser.add_argument("-train_folder", 
                           "--train_folder",
                          default = '/scratch/ramprasad.sa/probing_summarization_factuality/internal_states')

    argParser.add_argument("-annotator", 
                           "--annotator",
                          default = 'GPT_annotated')
    
    argParser.add_argument("-dataset", 
                           "--dataset",
                          default = 'XSUM')
    
    argParser.add_argument("-model", 
                           "--model",
                          default = 'mistral7b')
    
    argParser.add_argument("-prompt_type", 
                           "--prompt_type",
                          default = 'document_context_gpt')
    
    argParser.add_argument("-num_epochs", 
                           "--num_epochs",
                          default = 1000)
    
    argParser.add_argument("-learning_rate", 
                           "--learning_rate",
                          default = 0.001)
    
    argParser.add_argument("-test_folder", 
                           "--test_folder",
                          default = '/scratch/ramprasad.sa/probing_summarization_factuality/internal_states/Genaudit')
    
    argParser.add_argument("-write_dir", 
                           "--write_dir",
                          default = '/scratch/ramprasad.sa/probing_summarization_factuality/probes/linear_probe')
    
    args = argParser.parse_args()

    train_dir = f'{args.train_folder}/{args.annotator}/{args.dataset}/{args.model}/{args.prompt_type}'
    test_dir = f'{args.train_folder}/{args.annotator}/{args.dataset}/{args.model}/{args.prompt_type}'
    write_dir = f'{args.write_dir}/{args.annotator}/{args.dataset}/{args.model}/{args.prompt_type}'


    train_data, val_data, class_weights = HiddenStatesDataset().make_data(train_dir, 
                                                            hidden_state_idx = 32)

    test_data_1, test_data_2, _ = HiddenStatesDataset().make_data(test_dir, 
                                                            hidden_state_idx = 32)
    test_data = test_data_1 + test_data_2

    run_train(train_data = train_data,
              val_data = val_data,
              test_data = test_data,
              class_weights = class_weights,
              num_epochs = args.num_epochs,
              learning_rate = args.learning_rate,
              write_dir = write_dir)