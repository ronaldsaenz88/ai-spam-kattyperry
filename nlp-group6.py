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

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import utilities

import warnings
warnings.filterwarnings('ignore')
    
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

kattyperry_data_pre = kattyperry_data_prep


# Before the pre processing (cleaning data)

## Build a count vectorizer and extract term counts 
count_vectorizer0 = CountVectorizer(lowercase=True)
train_tc0 = count_vectorizer0.fit_transform(kattyperry_data_pre['CONTENT'])
print("\nDimensions of initial data:", train_tc0.shape)
#print(train_tc)

## Vocabulary
vocabulary_initial = np.array(count_vectorizer0.get_feature_names())
print("\nVocabulary Initial:\n", vocabulary_initial)


# Clean data
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT']
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_url(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_non_ascii(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_lowercase(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_contractions(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_punctuation(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_numbers(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_clean_stopwords(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_stem_words(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_lemmatize_nouns(x))
kattyperry_data_pre['CONTENT2'] = kattyperry_data_pre['CONTENT2'].apply(lambda x:utilities.fnc_lemmatize_verbs(x))
kattyperry_data_pre.head()


# After the pre processing (cleaning data)

## Build a count vectorizer and extract term counts 
count_vectorizer = CountVectorizer(lowercase=True)
train_tc = count_vectorizer.fit_transform(kattyperry_data_pre['CONTENT2'])
print("\nDimensions of vector data:", train_tc.shape)
#print(train_tc)

## Vocabulary
vocabulary_clean_data = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary of Clean Data:\n", vocabulary_clean_data)

#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
#type(train_tfidf)
#print(train_tfidf)
print("\nDimensions of the data:", train_tfidf.shape)
#print(train_tfidf.shape)

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
#print(train_tc_tr)

## Vocabulary
vocabulary_training = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary of training:\n", vocabulary_training)

#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
X_tr = train_tfidf_tr = tfidf.fit_transform(train_tc_tr)
Y_tr = Y_train
#type(train_tfidf_tr)
#print(train_tfidf_tr)

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
print("\nDimensions of testing data:", input_test.shape)
#type(input_test)
#print(input_test)

# Transform vectorized data using tfidf transformer
X_test_tfidf = input_test_tfidf = tfidf.transform(input_test)
#type(input_test_tfidf)
#print(input_test_tfidf)

# Predict the output categories
Y_test_pred = classifier.predict(X_test_tfidf)
#print(X_test_tfidf.shape)


accuracy = 100.0 * (Y_test == Y_test_pred).sum() / X_test_tfidf.shape[0]
print("\nAccuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

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
    'joIN ME',
    'Click here http://www.com',
    'You win the lottery',
    'love this song	',
    'I love you Katty',
    'Katty is the best singer around the world!!!!',
    'Free Katty Perry',
    'Help me Katty, I need money',
    'SUBSCRIBE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!	',
    'Check out this video on YouTube:..🌈🌈🌈'
]


# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)
#type(input_tc)
#print(input_tc)


# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
#type(input_tfidf)
#print(input_tfidf)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category, '\n', utilities.fnc_get_prediction_class(category))
        
