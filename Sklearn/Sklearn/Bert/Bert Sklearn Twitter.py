"""
BERT SKLEARN:

Produces 

https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.FunctionTransformer.html

https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

"""

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm  
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
import torch
import os

# stop words
nltk.download('stopwords')

#_________________CHECK GPU AVAILABILITY_______________________________
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Gets directory
cwd = os.getcwd()

#_________________LOAD BERT TO GPU_______________________________

#tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)


#_________________EMBEDDING TEXT (with GPU)_______________________________
def get_bert_embeddings(texts):
    # Tokenize all texts at once
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=8).to(device)
    
    with torch.no_grad():
        # Pass the inputs through BERT
        outputs = model(**inputs)

    # Average the token embeddings across the sequence length
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings


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
plt.xticks(ticks=np.arange(len(counts)), labels=['Negative', 'Positive'])
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

#___________________embedding _____________________
X_train_embeddings = get_bert_embeddings(X_train)
X_test_embeddings = get_bert_embeddings(X_test)

#____________________MODEL_________________________
# Train the classifier
classifier = LogisticRegression()

# ______________________________TRAINING__________________________________________

with tqdm(total=len(X_train), desc="Training model") as pbar:
    classifier.fit(X_train_embeddings, y_train)
    pbar.update(len(X_train_embeddings))


# ___________________________Predict__________________________________

#Predict
final_predictions = classifier.predict(X_test_embeddings)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, final_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the results
print('Accuracy: ', accuracy_score(y_test, final_predictions))
print('Precision: ', precision_score(y_test, final_predictions, average='macro'))
print('Recall: ', recall_score(y_test, final_predictions, average='macro'))
print('F1 Score: ', f1_score(y_test, final_predictions, average='macro'))
