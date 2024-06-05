import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import pandas as pd
import torch
import logging
import time
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, DistilBertTokenizer, DistilBertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

program_start_time = time.time()


log_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'logs/training_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
logging.basicConfig(filename=log_filename, filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)

def load_category_data(file_path: str, category: str):
    df = pd.read_feather(file_path)
    # add a column for the category
    df['category'] = category
    return df

model_map = {
  'bert-base-uncased': {
    'model': BertForSequenceClassification,
    'tokenizer': BertTokenizer,
  },
  'distilbert-base-uncased': {
    'model': DistilBertForSequenceClassification,
    'tokenizer': DistilBertTokenizer,
  },
  'distilbert-base-uncased-finetuned-sst-2-english': {
    'model': DistilBertForSequenceClassification,
    'tokenizer': DistilBertTokenizer,
  },
  'albert-base-v2': {
    'model': AlbertForSequenceClassification,
    'tokenizer': AlbertTokenizer,
  },
  'roberta-base': {
    'model': RobertaForSequenceClassification,
    'tokenizer': RobertaTokenizer,
  },
}

categories = {
  'crime': ['FA-KES-Dataset', 'snope'],
  'health': ['covid_claims', 'covid_fake_news_dataset', 'covid_FNIR'],
  'politics': ['politifact_dataset', 'liar_dataset', 'fake_news_dataset', 'pheme'],
  'science': ['climate_dataset'],
  'social_media': ['isot_dataset', 'gossipcop']
}


# show info on the categories and datasets
data = pd.concat([load_category_data(os.path.join(os.path.realpath('.'), f'.\data\{category}\{dataset}.feather'), category) for category, datasets in categories.items() for dataset in datasets])
print(data.groupby('category').count())


# keep only the text and category columns
data = data[['text', 'category']]
data.dropna(inplace=True)
print(data.groupby('category').count())


data['encoded_category'] = data['category'].astype('category').cat.codes
data.head()

# Match the number of samples for each category
df = data.groupby('category').apply(lambda x: x.sample(n=850, random_state=42))
df = df.reset_index(drop=True)
df.groupby('category').count()


df_texts = df['text'].to_list()
df_labels = df['encoded_category'].to_list()

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df_texts, df_labels, test_size=0.2, random_state=7623)


model_name = 'distilbert-base-uncased'
# load the tokenizer and model
tokenizer = model_map[model_name]['tokenizer'].from_pretrained(model_name)

# create the encodings
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


# Convert the encodings to PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels)

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_labels)

# Create a DataLoader for training and testing
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=8)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=8)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_inputs, train_masks, train_labels = train_inputs.to(device), train_masks.to(device), train_labels.to(device)
test_inputs, test_masks, test_labels = test_inputs.to(device), test_masks.to(device), test_labels.to(device)


iterations = 3
epochs_list = [13, 14]

def train_model(model, epochs):
    for epoch in range(epochs):  
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average train loss: {avg_train_loss:f}")
        logging.info(f"Epoch {epoch+1} average train loss: {avg_train_loss:f}")

for epochs in epochs_list:
    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_accuracy = 0
    best_g_mean = 0

    best_model = None
    for iteration in range(iterations):
        print(f'{epochs} epochs, iteration {iteration}')
        logging.info('====================================================')
        model = model_map[model_name]['model'].from_pretrained(model_name, num_labels=len(categories))
        model_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.info(f"Model: {model_name} {model_time} with {epochs} epochs")
        # Move the model and data to the GPU
        model.to(device)

        # Fine-tune the pre-trained BERT model
        train_start_time = time.time()
        train_model(model, epochs)
        train_end_time = time.time()

        # Evaluate the model
        eval_start_time = time.time()
        model.eval()
        predictions = []
        for batch in test_dataloader:
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': None}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        eval_end_time = time.time()

        # Calculate additional metrics
        precision = precision_score(test_labels.cpu(), predictions, average='macro')
        recall = recall_score(test_labels.cpu(), predictions, average='macro')
        f1 = f1_score(test_labels.cpu(), predictions, average='macro')
        accuracy = accuracy_score(test_labels.cpu(), predictions)
        g_mean = (recall*accuracy)**0.5

        # Log the results
        logging.info(f"{model_time} with {epochs} epochs: Evaluation Results:")
        logging.info(f"Training time: {train_end_time - train_start_time} seconds")
        logging.info(f"Inference time: {eval_end_time - eval_start_time} seconds")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F-score: {f1}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"G-mean: {g_mean}")

        # Print the evaluation results
        print(f"{model_name} {model_time} with {epochs} epochs: Evaluation Results:")
        print(f"Training time: {train_end_time - train_start_time} seconds")
        print(f"Inference time: {eval_end_time - eval_start_time} seconds")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-score: {f1}")
        print(f"Accuracy: {accuracy}")
        print(f"G-mean: {g_mean}")

        # Save the model if it is better than the last one    
        # Evaluate the model further by testing it with ones that are not in df but are in data
        # get the data that is not in df but is in data
        data_only = data[~data['text'].isin(df['text'])]

        data_only = data_only.groupby('category').apply(lambda x: x.sample(n=160, random_state=42, replace=True))

        # Load the data
        data_texts = data_only['text'].to_list()
        data_labels = data_only['encoded_category'].to_list()

        # Split the dataset into training and testing sets
        _, data_test_texts, _, data_test_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=7623)

        # Tokenize the data
        data_test_encodings = tokenizer(list(data_test_texts), truncation=True, padding=True)

        # Convert the encodings to PyTorch tensors

        data_test_inputs = torch.tensor(data_test_encodings['input_ids'])
        data_test_masks = torch.tensor(data_test_encodings['attention_mask'])

        # Create a DataLoader for testing

        data_test_data = TensorDataset(data_test_inputs, data_test_masks)
        data_test_dataloader = DataLoader(data_test_data, batch_size=8)

        # Evaluate the model
        eval_start_time = time.time()
        model.eval()
        predictions = []
        for batch in data_test_dataloader:
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': None}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        eval_end_time = time.time()

        # Calculate additional metrics
        precision = precision_score(data_test_labels, predictions, average='macro')
        recall = recall_score(data_test_labels, predictions, average='macro')
        f1 = f1_score(data_test_labels, predictions, average='macro')
        accuracy = accuracy_score(data_test_labels, predictions)
        g_mean = (recall*accuracy)**0.5

        # Log the results
        logging.info(f"{model_name} {model_time} with {epochs} epochs: Evaluation Results (completely new data):")
        logging.info(f"Training time: {train_end_time - train_start_time} seconds")
        logging.info(f"Inference time: {eval_end_time - eval_start_time} seconds")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F-score: {f1}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"G-mean: {g_mean}")

        # Print the evaluation results
        print(f"{model_name} {model_time} with {epochs} epochs: Evaluation Results (completely new data):")
        print(f"Training time: {train_end_time - train_start_time} seconds")
        print(f"Inference time: {eval_end_time - eval_start_time} seconds")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-score: {f1}")
        print(f"Accuracy: {accuracy}")
        print(f"G-mean: {g_mean}")


        # Save the model if it is better than the last one
        if precision > best_precision or recall > best_recall or f1 > best_f1 or accuracy > best_accuracy or g_mean > best_g_mean:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_accuracy = accuracy
            best_g_mean = g_mean
            best_model = model
            torch.save(model.state_dict(), f'classification/models/{model_name}-{model_time}.pt')
        else:
            del model
            torch.cuda.empty_cache()
            print(f"Model {model_name} {model_time} not saved")
            logging.info(f"Model {model_name} {model_time} not saved")


program_end_time = time.time()

logging.info(f"Total program time: {program_end_time - program_start_time} seconds")
print(f"Total program time: {program_end_time - program_start_time} seconds")