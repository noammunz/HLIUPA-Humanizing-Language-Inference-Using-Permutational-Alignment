import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, BartTokenizer, BartForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mnli_preprocessor import Preprocessor
from train import Trainer
import pickle as pkl
import sys
import time
import importlib
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

class Evaluator:
    def __init__(self, train_path, dev_matched_path, dev_mismatched_path, model_name, model_path, dev_permutations=20, seed=42):
        set_seed(seed)
        self.device = torch.device('cuda')
        if model_name == 'FacebookAI/roberta-large':
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        elif model_name == 'distilbert-base-uncased':
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        elif model_name == 'facebook/bart-large':
            self.model = BartForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path))
        
        preprocessor = Preprocessor(
            train_path,
            dev_matched_path,
            dev_mismatched_path,
            model_name=model_name,
            num_permutations=0,
            dev_permutations=dev_permutations,
            max_length=256,
            batch_size=16,
            seed=42,
            eval_only=True
        )

        self.train_dataloader, self.dev_matched_dataloader, self.dev_mismatched_dataloader = preprocessor.get_dataloaders()
        self.model.eval()
    

    def get_metrics(self, matched=True):
        accuracy, labels, predictions, ids, perm_ids = self.evaluate(matched=matched)
        df = self.create_df(labels, predictions, ids, perm_ids)
        omega_max = self.get_omega_max(df)
        omega_rand = self.get_omega_rand(df)
        pc = self.get_pc(df)
        pf = self.get_pf(df)
        return {
            'accuracy': accuracy,
            'omega_max': omega_max,
            'omega_rand': omega_rand,
            'pc': pc,
            'pf': pf
        }


    def evaluate(self, matched=True):
        if matched:
            dataloader = self.dev_matched_dataloader
        else:
            dataloader = self.dev_mismatched_dataloader
        
        predictions = []
        labels = []
        ids = []
        perm_ids = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                curr_labels = batch['labels'].to(self.device)
                curr_ids = batch['id']
                curr_perm_ids = batch['perm_id']

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(curr_labels.cpu().numpy())
                ids.extend([id.item() for id in curr_ids])
                perm_ids.extend([perm_id.item() for perm_id in curr_perm_ids])

                if i % 1000 == 0:
                    print(f'{i/len(dataloader):.2%} of evaluation complete...')

        accuracy = accuracy_score(labels, predictions)
        self.labels = labels
        self.predictions = predictions
        self.ids = ids
        self.perm_ids = perm_ids
        return accuracy, labels, predictions, ids, perm_ids
    

    def create_df(self, labels, predictions, ids, perm_ids):
        df = pd.DataFrame({
            'id': ids,
            'perm_id': perm_ids,
            'label': labels,
            'prediction': predictions
        })
        df.set_index(['id'], inplace=True)
        index_counts = df.index.value_counts()
        indexes_to_remove = index_counts[index_counts == 1]
        df = df[~df.index.isin(indexes_to_remove.index)]
        self.df = df
        return df
    

    def get_omega_max(self, df):
        df_copy = df.copy()
        unique_ids = df_copy.index.unique()
        omega_max = 0

        for id in unique_ids:
            curr_df = df_copy.loc[id]
            gold_label = curr_df.iloc[0]['label']
            for idx, row in curr_df.iloc[1:].iterrows():
                if row['prediction'] == gold_label:
                    omega_max += 1
                    break
        
        return omega_max / len(unique_ids)
    

    def get_omega_rand(self, df):
        df_copy = df.copy()
        unique_ids = df_copy.index.unique()
        omega_rand = 0

        for id in unique_ids:
            curr_df = df_copy.loc[id]
            gold_label = curr_df.iloc[0]['label']
            
            perm_df = curr_df.iloc[1:]
            correct_preds = len(perm_df[perm_df['prediction'] == gold_label])
            if (correct_preds / len(perm_df)) >= (1/3):
                omega_rand += 1
            
        return omega_rand / len(unique_ids)


    def get_pc(self, df):
        df_copy = df.copy()
        no_perms_df = df_copy[df_copy['perm_id'] == 0]
        correct_ids = no_perms_df[no_perms_df['label'] == no_perms_df['prediction']].index.unique()
        pc = 0

        for id in correct_ids:
            curr_df = df_copy.loc[id]
            gold_label = curr_df.iloc[0]['label']
            
            perm_df = curr_df.iloc[1:]
            correct_preds = len(perm_df[perm_df['prediction'] == gold_label])
            pc += correct_preds / len(perm_df)
        
        return pc / len(correct_ids)


    def get_pf(self, df):
        df_copy = df.copy()
        no_perms_df = df_copy[df_copy['perm_id'] == 0]
        incorrect_ids = no_perms_df[no_perms_df['label'] != no_perms_df['prediction']].index.unique()
        pf = 0

        for id in incorrect_ids:
            curr_df = df_copy.loc[id]
            gold_label = curr_df.iloc[0]['label']
            
            perm_df = curr_df.iloc[1:]
            correct_preds = len(perm_df[perm_df['prediction'] == gold_label])
            pf += correct_preds / len(perm_df)
        
        return pf / len(incorrect_ids)
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()