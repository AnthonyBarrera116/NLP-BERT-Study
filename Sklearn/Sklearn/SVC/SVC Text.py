"""
Linear SVC SKLEARN:

High results

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

# Hyperparameters
max_length = 256 

# stop words
nltk.download('stopwords')


# Gets directory
cwd = os.getcwd()
#_________________GET REVIEWS TEXT_______________________________

# Load dataset
data = pd.read_csv(cwd + '\models paper\SPAM text message 20170820 - Data.csv', encoding='latin-1')

# Renames to text and label
data = data.rename(columns={'Message': 'text', 'Category': 'label'})

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
plt.xticks(ticks=np.arange(len(counts)), labels=['ham', 'spam'])
plt.show()

#_________________CLEAN TEXT_______________________________

# Drop duplicates
data.drop_duplicates(inplace=True)

# one hot encode Positive and Negative
data['label'] = data['label'].map({'ham': 1, 'spam': 0})

# Clean Text
data['text'] = data['text']

#___________________TEST AND TRAIN SETS_____________________
# Split into x an y for train and test sets
x = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

X_train = X_train.astype(str).tolist()
X_test = X_test.astype(str).tolist()
#___________________TOKENIZE_____________________
# Tokenizer using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=256, stop_words='english')  
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = tfidf_vectorizer.transform(X_test).toarray()

# ______________________________MODELS__________________________________________
# Train SVM with a pipeline
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# ______________________________TRAINING__________________________________________
clf.fit(X_train_vectorized, y_train)


# ___________________________Predict__________________________________
final_predictions = clf.predict(X_test_vectorized)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, final_predictions), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the results with average='macro' for multiclass handling
print('Accuracy: ', accuracy_score(y_test, final_predictions))
print('Precision: ', precision_score(y_test, final_predictions, average='macro'))
print('Recall: ', recall_score(y_test, final_predictions, average='macro'))
print('F1 Score: ', f1_score(y_test, final_predictions, average='macro'))
