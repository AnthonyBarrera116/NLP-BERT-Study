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
    embeddings = []
    for text in tqdm(texts, desc="Encoding"):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Average the token embeddings
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embeddings)

#_________________CLEANING TEXT_______________________________

# Text cleaning function
def cleaning_text(text):

    text = text.lower()
    
    # Remove details
    #text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    #stop_words = set(stopwords.words('english'))

    # Rejoin text
    #text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text


#_________________GET REVIEWS TEXT_______________________________

# Load dataset
data = pd.read_csv(cwd + '\models paper\IMDB Dataset.csv', encoding='latin-1')

# Renames to text and label
data = data.rename(columns={'review': 'text', 'sentiment': 'label'})

# Only take label and text
data = data[["label", "text"]]

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
data['text'] = data['text'].apply(cleaning_text)

#___________________TEST AND TRAIN SETS_____________________
# Split into x an y for train and test sets
x = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

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
print('Precision: ', precision_score(y_test, final_predictions))
print('Recall: ', recall_score(y_test, final_predictions))
print('F1 Score: ', f1_score(y_test, final_predictions))