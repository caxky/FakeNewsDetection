import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import pandas as pd
import torch
import logging
import time
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, DistilBertTokenizer, DistilBertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

program_start_time = time.time()

balanced = True


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
    'tokenizer': BertTokenizerFast,
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

training_datasets = {
  'crime': ['FA-KES-Dataset'],
  'health': ['covid_FNIR', 'covid_fake_news_dataset'],
  'politics': ['pheme', 'liar_dataset', ],
  'science': ['climate_dataset'],
  'social_media': ['gossipcop']
}

testing_datasets = {
  'crime': ['snope'],
  'health': ['covid_claims'],
  'politics': ['politifact_dataset', 'fake_news_dataset'],
  'science': ['climate_dataset'],
  'social_media': ['isot_dataset']
}


# show info on the categories and datasets
training_data = pd.concat([load_category_data(os.path.join(os.path.realpath('.'), f'./data/{category}/{dataset}.feather'), category) for category, datasets in training_datasets.items() for dataset in datasets])
testing_data = pd.concat([load_category_data(os.path.join(os.path.realpath('.'), f'./data/{category}/{dataset}.feather'), category) for category, datasets in testing_datasets.items() for dataset in datasets])
print(f'Training data: {training_data.shape}')
print(f'Testing data: {testing_data.shape}')


# keep only the text and category columns
# keep only the text, category, and label columns
training_data = training_data[['text', 'category', 'label']]
training_data.dropna(inplace=True)
testing_data = testing_data[['text', 'category', 'label']]
testing_data.dropna(inplace=True)

# Match the number of samples for each category
df = training_data.groupby('label').apply(lambda x: x.sample(n=1000, random_state=42, replace=True)).groupby('category').apply(lambda x: x.sample(n=600, random_state=42, replace=True)) if balanced else training_data.apply(lambda x: x.sample(n=3000, random_state=42))
df = df.reset_index(drop=True)
logging.info(df.groupby('category').count())
print(df.groupby('category').count())
logging.info(df.groupby('label').count())
print(df.groupby('label').count())


df_texts = df['text'].to_list()
df_labels = df['label'].to_list()

# Split the dataset into training and validation sets
train_texts, validation_texts, train_labels, validation_labels = train_test_split(df_texts, df_labels, test_size=0.2, random_state=7623)


model_name = 'bert-base-uncased'
model_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.info('====================================================')
# load the tokenizer and model
tokenizer = model_map[model_name]['tokenizer'].from_pretrained(model_name)
model = model_map[model_name]['model'].from_pretrained(model_name, num_labels=2)

# create the encodings
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
validation_encodings = tokenizer(list(validation_texts), truncation=True, padding=True)


# Convert the encodings to PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels)

validation_inputs = torch.tensor(validation_encodings['input_ids'])
validation_masks = torch.tensor(validation_encodings['attention_mask'])
validation_labels = torch.tensor(validation_labels)

# Create a DataLoader for training and validation
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=8)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, batch_size=8)

# testing data
testing_df = testing_data.groupby('label').apply(lambda x: x.sample(n=1000, random_state=42, replace=True)).groupby('category').apply(lambda x: x.sample(n=160, random_state=42, replace=True))

# Load the data
testing_df_texts = testing_df['text'].to_list()
testing_df_labels = testing_df['label'].to_list()

# Tokenize the data
test_encodings = tokenizer(list(testing_df_texts), truncation=True, padding=True)


# Convert the encodings to PyTorch tensors
test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(testing_df_labels)

# Create a DataLoader for testing
test_data = TensorDataset(test_inputs, test_masks)
test_dataloader = DataLoader(test_data, batch_size=8)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_inputs, train_masks, train_labels = train_inputs.to(device), train_masks.to(device), train_labels.to(device)
validation_inputs, validation_masks, validation_labels = validation_inputs.to(device), validation_masks.to(device), validation_labels.to(device)
test_inputs, test_masks, test_labels = test_inputs.to(device), test_masks.to(device), test_labels.to(device)

iterations = 1
epochs_list = [1, 2, 3, 4, 5, 6, 7, 8]

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
        model = model_map[model_name]['model'].from_pretrained(model_name, num_labels=2)
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
        for batch in validation_dataloader:
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': None}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        eval_end_time = time.time()

        # Calculate additional metrics
        precision = precision_score(validation_labels.cpu(), predictions)
        recall = recall_score(validation_labels.cpu(), predictions)
        f1 = f1_score(validation_labels.cpu(), predictions)
        accuracy = accuracy_score(validation_labels.cpu(), predictions)
        g_mean = (recall*accuracy)**0.5

        # Log the results
        logging.info(f"{model_time} with {epochs} epochs: Evaluation Results {'(balanced)' if balanced else ''}:")
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
        precision = precision_score(test_labels.cpu(), predictions)
        recall = recall_score(test_labels.cpu(), predictions)
        f1 = f1_score(test_labels.cpu(), predictions)
        accuracy = accuracy_score(test_labels.cpu(), predictions)
        g_mean = (recall*accuracy)**0.5

        # Log the results
        logging.info(f"{model_name} {model_time} with {epochs} epochs: Evaluation Results {'(balanced)' if balanced else ''} (completely new data):")
        logging.info(f"Training time: {train_end_time - train_start_time} seconds")
        logging.info(f"Inference time: {eval_end_time - eval_start_time} seconds")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F-score: {f1}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"G-mean: {g_mean}")

        # Print the evaluation results
        print(f"{model_name} {model_time} with {epochs} epochs: Evaluation Results {'(balanced)' if balanced else ''} (completely new data):")
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
            torch.save(model.state_dict(), f'detection-general/models/{model_name}-{model_time}.pt')
        else:
            del model
            torch.cuda.empty_cache()
            print(f"Model {model_name} {model_time} not saved")
            logging.info(f"Model {model_name} {model_time} not saved")


program_end_time = time.time()

logging.info(f"Total program time: {program_end_time - program_start_time} seconds")
print(f"Total program time: {program_end_time - program_start_time} seconds")