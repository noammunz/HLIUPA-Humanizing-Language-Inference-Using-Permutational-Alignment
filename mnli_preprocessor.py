import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Preprocessor:
    """
    Preprocesses the MultiNLI dataset for training and evaluation.

    Args:
        train_path (str): The path to the training data.
        dev_matched_path (str): The path to the matched development data.
        dev_mismatched_path (str): The path to the mismatched development data.
        model_name (str): The name of the pretrained model to use.
        num_permutations (int): The number of unique permutations to generate for the training data.
        dev_permutations (int): The number of unique permutations to generate for the development data.
        config (str): The configuration for generating permutations. Can be '1', '2', or 'both'.
        max_length (int): The maximum length of the input sequences.
        batch_size (int): The batch size for the dataloaders.
        seed (int): The random seed to use.
    """
    def __init__(self, train_path, dev_matched_path, dev_mismatched_path, model_name,
                    num_permutations=5, dev_permutations=5, config='both', max_length=256, batch_size=16, seed=42):
        self.train_path = train_path
        self.dev_matched_path = dev_matched_path
        self.dev_mismatched_path = dev_mismatched_path
        self.model_name = model_name
        self.num_permutations = num_permutations
        self.config = config
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.num_permutations = num_permutations
        self.dev_permutations = dev_permutations
        self.config = config
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        set_seed(seed)
    
    def load_data(self):
        """
        Loads the training and development data.

        Returns:
            pd.DataFrame: The training data.
        """
        train_df = pd.read_json(self.train_path, lines=True)
        dev_matched_df = pd.read_json(self.dev_matched_path, lines=True)
        dev_mismatched_df = pd.read_json(self.dev_mismatched_path, lines=True)
        return train_df, dev_matched_df, dev_mismatched_df
    
    def tokenize_data(self, df, batch_size=2048):
        input_ids = []
        attention_masks = []

        for i in tqdm(range(0, len(df), batch_size), desc='Tokenizing data'):
            batch = df[i:i+batch_size]
            encodings = self.tokenizer.batch_encode_plus(
                list(zip(batch['sentence1'], batch['sentence2'])),
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encodings['input_ids'])
            attention_masks.append(encodings['attention_mask'])
    
        return {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attention_masks),
            'labels': torch.tensor(df['label']),
            'id': df['id'],
            'perm_id': df['perm_id']
        }

    def jumble_words(self, text, n):
        """
        Jumbles the words in a sentence n times.

        Args:
            text (str): The sentence to jumble.
            n (int): The number of unique permutations to generate.

        Returns:
            list: A list of unique permutations of the input sentence.
        """
        words = text.split()
        result = []
        attempts = n * 2
        
        while len(result) < n and attempts > 0:
            shuffled = words.copy()
            random.shuffle(shuffled)
            jumbled = ' '.join(shuffled)
            if jumbled not in result:
                result.append(jumbled)
            attempts -= 1
        
        return result

    def jumble_dataframe(self, df, n):
        """
        Jumbles the data in the dataframe n times.

        Args:
            df (pd.DataFrame): The dataframe to jumble.
            n (int): The number of unique permutations to generate.

        Returns:
            pd.DataFrame: The jumbled dataframe.
        """
        jumbled_data = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc='Jumbling data'):
            original_row = row.to_dict() 
            original_row['perm_id'] = 0
            jumbled_data.append(original_row)
            max_perms1 = min(n, len(row['sentence1'].split()) - 1)
            max_perms2 = min(n, len(row['sentence2'].split()) - 1)

            unique_permutations1 = self.jumble_words(row['sentence1'], max_perms1) if self.config in ['1', 'both'] else [row['sentence1']] * n
            unique_permutations2 = self.jumble_words(row['sentence2'], max_perms2) if self.config in ['2', 'both'] else [row['sentence2']] * n

            for i, (perm1, perm2) in enumerate(zip(unique_permutations1, unique_permutations2)):
                new_row = original_row.copy()
                new_row['perm_id'] = i + 1
                new_row['sentence1'] = perm1
                new_row['sentence2'] = perm2
                jumbled_data.append(new_row) 
            
        return pd.DataFrame(jumbled_data)

    def get_tokens(self, df, n):
        """
        Tokenizes the data in the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to tokenize.
            n (int): The number of unique permutations to generate.

        Returns:
            dict: A dictionary containing the tokenized data.
        """
        df = df[df['gold_label'].isin(self.label_map.keys())].reset_index(drop=True)
        df.loc[:, 'label'] = df['gold_label'].map(self.label_map)
        df.loc[:, 'id'] = df.index
        df = df[['id', 'sentence1', 'sentence2', 'label']]
        df = self.jumble_dataframe(df, n)
        return self.tokenize_data(df)
    
    def get_dataloaders(self):
        """
        Returns the training and development dataloaders.
        
        Returns:
            DataLoader: The training dataloader.
            DataLoader: The matched development dataloader.
            DataLoader: The mismatched development dataloader.
        """
        train_df, dev_matched_df, dev_mismatched_df = self.load_data()
        train_tokens = self.get_tokens(train_df, self.num_permutations)
        dev_matched_tokens = self.get_tokens(dev_matched_df, self.dev_permutations)
        dev_mismatched_tokens = self.get_tokens(dev_mismatched_df, self.dev_permutations)

        train_dataset = NLIDataset(train_tokens)
        dev_matched_dataset = NLIDataset(dev_matched_tokens)
        dev_mismatched_dataset = NLIDataset(dev_mismatched_tokens)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_matched_loader = DataLoader(dev_matched_dataset, batch_size=self.batch_size, shuffle=False)
        dev_mismatched_loader = DataLoader(dev_mismatched_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, dev_matched_loader, dev_mismatched_loader

class NLIDataset(Dataset):
    """
    A PyTorch Dataset for the MultiNLI dataset.

    Args:
        tokens (dict): A dictionary containing the tokenized data.
    """
    def __init__(self, tokens):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = tokens['labels']
        self.id = tokens['id']
        self.perm_id = tokens['perm_id']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'id': self.id[idx],
            'perm_id': self.perm_id[idx]
        }