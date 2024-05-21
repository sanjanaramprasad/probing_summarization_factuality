import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HiddenStatesDataset(Dataset):
    def __init__(self):
        # train_data, test_data = self.make_data(folder_path,
        #                hidden_state_idx,
        #                padding)
        # self.train_data = train_data
        # self.test_data = test_data
        
        
        return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.summary_hidden_states[idx], self.summary_token_labels[idx], self.summary_tokens[idx]
        return self.data[idx]
     
    def make_padded_data(self, 
                         summary_hidden_states, 
                         summary_tokens,
                         summary_token_labels):
        
        padded_hidden_states = []
        padded_token_labels = []
        padded_tokens = []

        for hidden_state, token, token_label in zip(summary_hidden_states, summary_tokens, summary_token_labels):
            num_pad = self.pad_dimension - hidden_state.shape[0]
            padding_tensor = torch.tensor([-1] * num_pad)
            padded_hidden_states.append(torch.cat([hidden_state, torch.zeros(num_pad, self.hidden_state_dimension)]))
            padded_token_labels.append(torch.cat([token_label, padding_tensor]))
            padded_tokens.append(torch.cat([token, padding_tensor]))

        return padded_hidden_states, padded_tokens, padded_token_labels

    def load_hstate_folder(self, 
                           folder_path, 
                           hidden_state_idx):
        
        files = os.listdir(folder_path)
        # print(files[:10])
        all_summary_hidden_states = []
        all_summary_token_labels = []
        all_summary_tokens = []

        for file_name in tqdm(files):
            if os.path.isfile(os.path.join(folder_path, file_name)):
                
                example = torch.load(os.path.join(folder_path, file_name))
                # print(example)
                source_len = example['source_len']
                summary_len = example['summary_len']
                hidden_states = example['hidden_states']
                summary_tokens = example['all_tokens'][source_len: source_len + summary_len]
                summary_token_labels = example['summary_token_labels']
                summary_hidden_states = hidden_states[:, source_len - 1: (source_len + summary_len) - 1, :]
                assert summary_tokens.shape[0] == len(summary_token_labels) == summary_hidden_states.shape[1]
                
                #### if factual label in summary
                # if 1 in summary_token_labels:
                if True:
                    all_summary_hidden_states.append(summary_hidden_states[hidden_state_idx])
                    all_summary_token_labels.append(torch.tensor(summary_token_labels))
                    all_summary_tokens.append(summary_tokens)
        return all_summary_hidden_states, all_summary_tokens, all_summary_token_labels

    def calculate_class_weights(self, data):
        all_labels = [datapoint[2][datapoint[2]!= -1] for datapoint in data]
        num_nonfactual = len([labels for labels in all_labels if 1 in labels])
        num_factual = len([labels for labels in all_labels if 1 not in labels])

        print(num_nonfactual, num_factual)
        total_samples = len(data)
        num_classes = 2
        weight_factual = 1
        weight_nonfactual= 1
        if num_nonfactual != 0 and num_factual != 0:
            weight_factual = total_samples / (num_classes * num_factual)
            weight_nonfactual = total_samples / (num_classes * num_nonfactual)
        class_weights = torch.tensor([weight_nonfactual, weight_factual])
        class_weights = {0 : weight_factual, 1:  weight_nonfactual}
        return class_weights


    def make_train_test_splits(self, data):
        factual_data = []
        nonfactual_data = []
        for datapoint in data:
            labels = datapoint[2]
            labels = labels[labels != -1]
            if torch.sum(labels == 1).item() >0 :
                nonfactual_data.append(datapoint)
            else:
                factual_data.append(datapoint)

        train_data = []
        test_data = []
        for subset in [factual_data, nonfactual_data]:
            if subset:
                train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)
                train_data += train_subset
                test_data += test_subset
            
        return train_data, test_data

    def make_data(self, folder_path, hidden_state_idx, padding=True):
        
        summary_hidden_states, summary_tokens, summary_token_labels = self.load_hstate_folder(folder_path, hidden_state_idx)
        if padding:
            hidden_state_dimensions = set(hidden_state.shape[1] for hidden_state in summary_hidden_states)
            assert len(hidden_state_dimensions) == 1
            self.hidden_state_dimension = hidden_state_dimensions.pop()
            self.pad_dimension = max(hidden_state.shape[0] for hidden_state in summary_hidden_states)

            summary_hidden_states, summary_tokens, summary_token_labels = self.make_padded_data(
                summary_hidden_states, summary_tokens, summary_token_labels)
            
        data = list(zip(summary_hidden_states, summary_tokens, summary_token_labels))
        
        
        train_data, test_data = self.make_train_test_splits(data)
        class_weights = self.calculate_class_weights(train_data)
        return train_data, test_data, class_weights

