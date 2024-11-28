"""
Mulitniomial NB SKLEARN:

High results

https://scikit-learn.org/0.24/modules/generated/sklearn.naive_bayes.MultinomialNB.html 

""" 

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm  
from sklearn.naive_bayes import MultinomialNB
import os

# Hyperparameters
max_length = 256 

# stop words
nltk.download('stopwords')


# Gets directory
cwd = os.getcwd()
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


# ______________________________MODELS__________________________________________

model = MultinomialNB()

# ______________________________TRAINING__________________________________________

model.fit(X_train_vectorized, y_train)

with tqdm(total=X_train_vectorized.shape[1], desc="Training models") as pbar:
    model.fit(X_train_vectorized, y_train)
    pbar.update(X_train_vectorized.shape[1])

# ___________________________Predict__________________________________
final_predictions = model.predict(X_test_vectorized)

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