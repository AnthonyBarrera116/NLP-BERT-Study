"""
BERT MODEL:

WILL RUN AND PRODUCES LOW RESULTS

LEARNED FROM DEEP LEARNING CLASS TAHT I AM IN NOW
additional:
https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b

"""
# Libraries
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel,get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import os

# Gets directory
cwd = os.getcwd()

# Hyperparameters
bert_model_name = 'bert-large-uncased'
num_classes = 2
max_length = 128
batch_size = 64
num_epochs = 4
learning_rate = 0.001

# Cuda setup 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available()) 
print("CUDA device count:", torch.cuda.device_count()) 
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))


#_________________BERT TOKENIZER MODEL_______________________________
# Dataset encoding tokeninzing class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):

        # Self variables for class
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Length of texts
    def __len__(self):

        return len(self.texts)

    # Tokeninizing and encoding text and label 
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label, dtype=torch.long)}


#_________________BERT MODEL_______________________________
class BERTClassifier(nn.Module):

    # Building of model
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    # Layers for input
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

#_________________GET REVIEWS TEXT_______________________________

# Load dataset
data = pd.read_csv(cwd + '\models paper\comments.csv', encoding='latin-1')

# Renames to text and label
data = data.rename(columns={'im getting on borderlands and i will murder you all ,': 'text', 'Positive': 'label'})

data = data[data['label'] != 'Irrelevant']

data = data[data['label'] != 'Neutral']

print(data)

#___________________Distubution Between positive and Negative_________________

# counts # of pos and neg
counts = data['label'].value_counts()
print("Class counts:", counts) 

# Model of count between both
plt.figure(figsize=(6, 4))
sns.barplot(x=counts.index.astype(str), y=counts.values, palette='Set1') 
plt.title('Distribution of Spam vs Ham')
plt.xlabel('Type of message')
plt.ylabel('Count')
plt.xticks(ticks=np.arange(len(counts)), labels=['negative', 'positive'])
plt.show()

#_________________CLEAN TEXT_______________________________

# Drop duplicates
data.drop_duplicates(inplace=True)

# one hot encode Positive and Negative
data['label'] = data['label'].map({'positive': 1, 'negative': 0})

# Clean Text
data['text'] = data['text']


#___________________TEST AND TRAIN SETS_____________________
# Split into x an y for train and test sets
x = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#___________________TOKENIZE_____________________

# Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Datasets
train_dataset = TextClassificationDataset(X_train.reset_index(drop=True), y_train.reset_index(drop=True), tokenizer, max_length)
test_dataset = TextClassificationDataset(X_test.reset_index(drop=True), y_test.reset_index(drop=True), tokenizer, max_length)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#________________________Model___________________________________________
# Initialize the BERT model for classification
model = BERTClassifier(bert_model_name, num_classes).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

#________________________Training___________________________________________
# Training loop
for epoch in range(num_epochs):

    # Epoch on
    print(f"Epoch {epoch + 1}/{num_epochs}")
  
    # Train model
    model.train()
    
    # Loss per epoch
    epoch_loss = 0

    # Progress Bar
    progress_bar = tqdm(total=len(train_dataset), desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')
    
    # Gets each batch
    for batch in train_dataloader:

        # Optimizer to zero
        optimizer.zero_grad()

        # input and leels
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # preprocess data before bert model
        attention_mask = batch['attention_mask'].to(device)

        # Output for each thing in batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Loss from type of loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Adds epoch loss for total epoch loss
        epoch_loss += loss.item()

        # updates bar and empties cache
        progress_bar.update(batch_size)
        progress_bar.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

    # CLose bar
    progress_bar.close()


# Save the trained model
torch.save(model.state_dict(), "bert_classifier.pth")

#________________________evaluate______________________________________
# Evaluate the model
model.eval()

# Predictions and actual labels
predictions = []
actual_labels = []

# Testing
with torch.no_grad():
    
    # Test in batches
    for batch in test_dataloader: 

        # Load test batch text and label using the correct keys
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Prediction
        _, preds = torch.max(outputs, dim=1)

        # Append to prediction and actual label lists 
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())


# Confusion matrix
confusion_mat = confusion_matrix(actual_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the results
print('Accuracy: ', accuracy_score(y_test.values, predictions))
print('Precision: ', precision_score(y_test.values, predictions, average='weighted'))
print('Recall: ', recall_score(y_test.values, predictions, average='weighted'))
print('F1 Score: ', f1_score(y_test.values, predictions, average='weighted'))
print(classification_report(y_test.values, predictions))