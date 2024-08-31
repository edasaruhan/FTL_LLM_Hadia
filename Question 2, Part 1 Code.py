# Install necessary libraries
!pip install transformers torch sklearn gensim faiss-cpu

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.downloader import load
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load the IMDB dataset
from sklearn.datasets import fetch_openml

data = fetch_openml('imdb', version 1)
texts = data.data['text']
labels = data.target
labels = (labels == 'positive').astype(int)  # Convert to binary labels

# Preprocess the text and split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define functions for embeddings
def get_word2vec_embeddings(texts, model):
    embeddings = []
    for text in texts:
        words = text.split()
        word_vecs = [model[word] for word in words if word in model]
        if word_vecs:
            embeddings.append(np.mean(word_vecs, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)

def get_glove_embeddings(texts, model):
    embeddings = []
    for text in texts:
        words = text.split()
        word_vecs = [model[word] for word in words if word in model]
        if word_vecs:
            embeddings.append(np.mean(word_vecs, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)

class BERTEmbedder:
    def _init_(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def embed(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

# Define the classification model
class SimpleNN(nn.Module):
    def _init_(self, input_dim):
        super(SimpleNN, self)._init_()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model(train_data, train_labels, input_dim):
    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                                   torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(5):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_model(model, test_data, test_labels):
    with torch.no_grad():
        predictions = model(torch.tensor(test_data, dtype=torch.float32)).squeeze().numpy()
        predictions = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    return accuracy, precision, recall, f1

# Load word2vec and GloVe models
word2vec_model = load('word2vec-google-news-300')
glove_model = load('glove-wiki-gigaword-100')

# Create embeddings
word2vec_train = get_word2vec_embeddings(X_train, word2vec_model)
word2vec_test = get_word2vec_embeddings(X_test, word2vec_model)

glove_train = get_glove_embeddings(X_train, glove_model)
glove_test = get_glove_embeddings(X_test, glove_model)

bert_embedder = BERTEmbedder()
bert_train = bert_embedder.embed(X_train)
bert_test = bert_embedder.embed(X_test)

# Train and evaluate models
def run_experiment(train_data, test_data, train_labels, test_labels, input_dim):
    model = train_model(train_data, train_labels, input_dim)
    accuracy, precision, recall, f1 = evaluate_model(model, test_data, test_labels)
    return accuracy, precision, recall, f1

word2vec_results = run_experiment(word2vec_train, word2vec_test, y_train, y_test, word2vec_train.shape[1])
glove_results = run_experiment(glove_train, glove_test, y_train, y_test, glove_train.shape[1])
bert_results = run_experiment(bert_train, bert_test, y_train, y_test, bert_train.shape[1])

# Print results
print("word2vec Results:")
print(f"Accuracy: {word2vec_results[0]}")
print(f"Precision: {word2vec_results[1]}")
print(f"Recall: {word2vec_results[2]}")
print(f"F1 Score: {word2vec_results[3]}")

print("\nGloVe Results:")
print(f"Accuracy: {glove_results[0]}")
print(f"Precision: {glove_results[1]}")
print(f"Recall: {glove_results[2]}")
print(f"F1 Score: {glove_results[3]}")

print("\nBERT Results:")
print(f"Accuracy: {bert_results[0]}")
print(f"Precision: {bert_results[1]}")
print(f"Recall: {bert_results[2]}")
print(f"F1 Score: {bert_results[3]}")