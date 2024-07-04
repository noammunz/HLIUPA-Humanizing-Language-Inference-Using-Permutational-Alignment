import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, BartTokenizer, BartForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np
import random
import sys
import time
from peft import get_peft_model, LoraConfig, TaskType

import logging
logging.disable(logging.WARNING)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def printf(text, start_time, verbose=True):
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    elapsed_timestamp = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    print(f'{elapsed_timestamp}    {text}')
    sys.stdout.flush()

class Trainer:
    def __init__(
            self, model_name, num_labels, train_dataloader, dev_matched_dataloader, dev_mismatched_dataloader,
            device='cuda', lora=False, epochs=2, batch_size=16, learning_rate=1e-5, seed=42, verbose=True
    ):
        set_seed(seed)
        self.device = torch.device(device)
        if model_name == 'FacebookAI/roberta-large':
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        elif model_name == 'distilbert-base-uncased':
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        elif model_name == 'facebook/bart-large':
            self.model = BartForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.train_dataloader = train_dataloader
        self.dev_matched_dataloader = dev_matched_dataloader
        self.dev_mismatched_dataloader = dev_mismatched_dataloader
        self.start_time = time.time()
        self.verbose = verbose

        if lora:
            peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=2,  # rank of the update matrices
            lora_alpha=8,  # alpha scaling factor
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "key", "value"]  # Apply LoRA to attention modules
            )
            self.model = get_peft_model(self.model, peft_config).to(self.device)

    def train(self, eval=True):
        self.start_time = time.time()
        printf(f'Starting training...', self.start_time, self.verbose)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        losses = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for i, batch in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
                if i % 300 == 0:
                    progress = ((epoch * len(self.train_dataloader)) + i + 1) / (self.epochs * len(self.train_dataloader))
                    progress_str = f'{progress:.2%}'
                    printf(f'({progress_str})  Epoch {epoch+1}, Batch {i+1}/{len(self.train_dataloader)}, Loss: {train_loss/(i+1)}', self.start_time, self.verbose)
            
            if eval:
                matched_accuracy, _, _, _, _ = self.evaluate(matched=True)
                mismatched_accuracy, _, _, _, _ = self.evaluate(matched=False)
                printf(f'Epoch {epoch+1}, Matched Accuracy: {matched_accuracy}, Mismatched Accuracy: {mismatched_accuracy}', self.start_time, self.verbose)
        
        self.losses = losses
        printf(f'Training complete!', self.start_time, self.verbose)

    def evaluate(self, matched=True):
        self.model.eval()
        predictions = []
        labels = []
        ids = []
        perm_ids = []

        if matched:
            dataloader = self.dev_matched_dataloader
        else:
            dataloader = self.dev_mismatched_dataloader

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                curr_labels = batch['labels'].to(self.device)
                curr_ids = batch['id']
                curr_perm_ids = batch['perm_id']

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(curr_labels.cpu().numpy())
                ids.extend(curr_ids)
                perm_ids.extend(curr_perm_ids)
        
        accuracy = accuracy_score(labels, predictions)
        return accuracy, labels, predictions, ids, perm_ids

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)