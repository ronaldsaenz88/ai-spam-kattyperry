# -*- coding: utf-8-sig -*-
"""
Created on Thu Apr  7 22:36:46 2022

@author: Group 6
Ronald Saenz
Rudy Huayhua
Albert Bota
"""

import os
import pandas as pd
import numpy as np

# Import nltk packages and Punkt Tokenizer Models
import nltk
nltk.download("punkt")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import re, string, unicodedata
import contractions
import inflect

#remove the punctuations and stopwords
'''
Removing punctuations like . , ! $( ) * % @
Removing URLs
Removing Stop words
Lower casing
Tokenization
Stemming
Lemmatization
'''
import string
def text_process(text):
    stop_words = set(stopwords.words('english'))
    #text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokenized_words = nltk.word_tokenize(text)
    text = [w for w in tokenized_words if not w.lower() in stop_words] 
    # text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return "".join(new_words)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return "".join(new_words)

def remove_punctuation2(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_words.append(new_word)
    return "".join(new_words)

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word+" ")
        else:
            new_words.append(word)
    return "".join(new_words)


def find(x):
    if x == 1:
        return "Message is SPAM"
    else:
        return "Message is NOT Spam"
    
    
#############################################################################################################################
# a. Get the data 
#############################################################################################################################

'''
1. Load the data into a pandas data frame.
'''

# Input file containing data
print('\n**** Get the data  ****')

path = os.path.realpath('.')
filename = 'Youtube02-KatyPerry5.csv'
#filename = 'prueba2.csv'
fullpath = os.path.join(path,filename)
kattyperry_data = pd.read_csv(fullpath, sep=',')
print(kattyperry_data)
        

#############################################################################################################################
# b. Initial Exploration
#############################################################################################################################

'''
2. Carry out some basic data exploration and present your results.
     (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)
'''

print('\n**** Initial Exploration ****')

print('\nPrint the first 3 records')
print(kattyperry_data.head(3))

print('\nPrint the shape of the dataframe')
print(kattyperry_data.shape)

print('\nDisplay (print) the names, types and counts (showing  missing values per column')
print(kattyperry_data.info())


for column in kattyperry_data:
    print("Column:", column, " - Len:", len(kattyperry_data[column].unique()))
    
print("Unique values - AUTHOR Column: ", kattyperry_data["AUTHOR"].unique())
print("Unique values - CLASS Column: ", kattyperry_data["CLASS"].unique())
print("Unique values - CONTENT Column: ", kattyperry_data["CONTENT"].unique())

'''
Column: COMMENT_ID  - Len: 350
Column: AUTHOR  - Len: 342
Column: DATE  - Len: 350
Column: CONTENT  - Len: 348
Column: CLASS  - Len: 2

delete COMMENT_ID and DATE 
delete AUTHOR
Keep   CONTENT and CLASS
'''

print(kattyperry_data['CLASS'].value_counts())


## Delete columns
kattyperry_data_prep = kattyperry_data.drop(columns=["COMMENT_ID", "DATE", "AUTHOR"])


#############################################################################################################################
# c. Data Preparation 
#############################################################################################################################

'''
3. Using nltk toolkit classes and methods prepare the data for model building, 
    refer to the third lab tutorial in module 11 (Building a Category text predictor ). 
    Use count_vectorizer.fit_transform().
    
4. Present highlights of the output (initial features) such as the new 
    shape of the data and any other useful information before proceeding.

5. Downscale the transformed data using tf-idf and again present highlights of the output (final features) 
    such as the new shape of the data and any other useful information before proceeding.

'''

#storing the puntuation free text
kattyperry_data_pre = kattyperry_data_prep
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT']

kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:remove_non_ascii(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:replace_contractions(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:remove_punctuation2(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:to_lowercase(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:replace_numbers(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:text_process(x))
kattyperry_data_pre.head()

## Build a count vectorizer and extract term counts 
count_vectorizer = CountVectorizer(lowercase=True)
train_tc = count_vectorizer.fit_transform(kattyperry_data_pre['CONTENT2'])
print("\nDimensions of training data:", train_tc.shape)

print(train_tc)

## Vocabulary
vocabulary = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary)

#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)
print(train_tfidf)
print(train_tfidf.shape)

X = train_tfidf
Y = kattyperry_data_pre['CLASS']

dfeatures = pd.DataFrame(train_tfidf)
dfeatures['content'] = kattyperry_data_pre['CONTENT2']
dfeatures['target'] = kattyperry_data_pre['CLASS']
dfeatures['vector'] = dfeatures[0]

#############################################################################################################################
# d. Fit the training data 
#############################################################################################################################

'''
6. Use pandas.sample to shuffle the dataset, set frac =1 
7. Using pandas split your dataset into 75% for training and 25% for testing, 
    make sure to separate the class from the feature(s). (Do not use test_train_ split)
8. Fit the training data into a Naive Bayes classifier. 
'''

train_size = round(0.75 * dfeatures.shape[0])
dfeatures_sample = dfeatures.sample(frac=1, random_state=1)
dfeatures_training = dfeatures_sample.iloc[0:train_size-1] 
dfeatures_test = dfeatures_sample.iloc[train_size:]

print(f"No. of training examples: {dfeatures_training.shape[0]}")
print(f"No. of testing examples: {dfeatures_test.shape[0]}")

X_train = dfeatures_training['content']
Y_train = dfeatures_training['target']
X_test = dfeatures_test['content']
Y_test = dfeatures_test['target']

## Build a count vectorizer and extract term counts 
count_vectorizer2 = CountVectorizer(lowercase=True)
train_tc_tr = count_vectorizer.fit_transform(X_train)
print("\nDimensions of training data:", train_tc_tr.shape)
print(train_tc_tr)

## Vocabulary
vocabulary3 = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary3)

#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
X_tr = train_tfidf_tr = tfidf.fit_transform(train_tc_tr)
Y_tr = Y_train
type(train_tfidf_tr)
print(train_tfidf_tr)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(X_tr, Y_tr)


#############################################################################################################################
# E. VALIDATE MODEL 
#############################################################################################################################

'''
9. Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
'''

# Transform input data using count vectorizer
input_test = count_vectorizer.transform(X_test)
type(input_test)
print(input_test)

# Transform vectorized data using tfidf transformer
X_test_tfidf = input_test_tfidf = tfidf.transform(input_test)
type(input_test_tfidf)
print(input_test_tfidf)

# Predict the output categories
Y_test_pred = classifier.predict(X_test_tfidf)
print(X_test_tfidf.shape)


accuracy = 100.0 * (Y_test == Y_test_pred).sum() / X_test_tfidf.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

###############################################
# Cross validate - Scoring functions 

num_folds = 5
accuracy_values = cross_val_score(classifier, X, Y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier, X, Y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier, X, Y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, X, Y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")

###############################################
# Confusion Matrix

# Checking Classification Results with Confusion Matrix
# Naive Bayes
y_pred_nb = classifier.predict(X_test_tfidf)
y_true_nb = Y_test
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()


#############################################################################################################################
# F. TEST MODEL 
#############################################################################################################################

'''
10. Test the model on the test data, print the confusion matrix and the accuracy of the model.
11. As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam)
     and pass them to the classifier and check the results. 
     You can be very creative and even do more happy with light skin tone emoticon.
'''

# Define test data 
input_data = [
    'You need to be careful with cars when you are driving on slippery roads', 
    'A lot of devices can be operated wirelessly',
    'Players need to be careful when they are close to goal posts',
    'debates us understand the perspectives of both sides',
    'A feel pain every morning',
    'I got a speed ticket this morning comming to work',
    'I study a lot to improve my life',
    'Barcelona won its game',
    'Click here',
    'www.facebook.com/asfdfd',
    'joIN ME',
    'hOLA SOY RONALD',
    'free money',
    'FRee ',
    'Click http://www.com',
    'YOu win the lottery',
    'love this song﻿	',
    'I love you Katty',
    'Katty perry You are the best',
    'Katty you rock',
    'Katty is the best singer around the world!!!!',
    'Free Katty',
    'Help me Katty',
    'Join me Katty',
    'Help me Katty, I need money',
    'Check out my music niggas﻿',
    'SUBSCRIBE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!﻿	',
    'Check out this video on YouTube:..🌈🌈🌈',
    'Party rock﻿'
]


# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)
type(input_tc)
print(input_tc)

# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print(input_tfidf)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category, '\n', find(category))
        
