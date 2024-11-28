"""
Linear SVC MODEL:

WILL RUN AND PRODUCES LOW RESULTS

LEARNED FROM DEEP LEARNING CLASS TAHT I AM IN NOW

"""
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import os 

# Gets directory
cwd = os.getcwd()

# Cuda setup 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available()) 
print("CUDA device count:", torch.cuda.device_count()) 
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Hyperparameters
num_classes = 2
max_length = 256 
batch_size = 16
num_epochs = 5
learning_rate = 0.001

# Stop words setup
nltk.download('stopwords')

#__________________________LinearSVC__________________________________________
class LinearSVC(nn.Module):
    def __init__(self, input_dim):
        super(LinearSVC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
    

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
data['label'] = data['label'].map({'Positive': 1, 'Negative': 0})

# Clean Text
data['text'] = data['text']

data = data.dropna(subset=['label'])

#___________________TEST AND TRAIN SETS_____________________
# Split into x an y for train and test sets
x = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

X_train = X_train.astype(str).tolist()
X_test = X_test.astype(str).tolist()

#___________________TOKENIZE_____________________
# Tokenizer
tfidf_vectorizer = TfidfVectorizer(max_features=256)  
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = tfidf_vectorizer.transform(X_test).toarray()

# Dataloaders
train_dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_train_vectorized), torch.LongTensor(y_train.values)), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(TensorDataset(torch.FloatTensor(X_test_vectorized), torch.LongTensor(y_test.values)), batch_size=batch_size, shuffle=True)

#________________________MOdel___________________________________________
model = LinearSVC(input_dim=X_train_vectorized.shape[1]).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

#________________________Training___________________________________________
# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    
    epoch_loss = 0
    progress_bar = tqdm(total=len(X_train), desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Access the inputs and labels from the batch
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # epoch loss
        epoch_loss += loss.item()

        # Progress bar update and empty cuda Cache
        progress_bar.update(batch_size)
        progress_bar.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()
    
    # close bar
    progress_bar.close()
    print(f"Loss: {epoch_loss / len(train_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "linear_svc_classifier.pth")


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

        # load test batch text and label
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Outputs
        outputs = model(inputs)

        # Prediction
        _, preds = torch.max(outputs, dim=1)

        # append to prediction and actual list 
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

