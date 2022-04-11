# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:08:12 2022

@author: rhuay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



######

# 06/04/22
#1 Load the data into a pandas data frame.
#2 Carry out some basic data exploration and present your results. (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)

# File path
path = "C:/Users/rhuay/Documents/CENTENNIAL/AI/PROJECT"
filename = 'Youtube02-KatyPerry.csv'
fullpath = os.path.join(path,filename)

# Read data
data = pd.read_csv(fullpath)

# --------------------------------- Initial Exploration ----------------------------------#
# First 3 records
print(data.head(3))
print(data.shape)
# Names columns
print(data.columns.values)
# types
print(data.dtypes) 
# count of values
data.describe()

# remome AUTHOR, DATE, COMMENT_ID
dfinal = data.drop(columns=['AUTHOR','DATE','COMMENT_ID'])
dfinal.columns.values
dfinal.dtypes
dfinal.head(3)


# 07/04/22
#3 Using nltk toolkit classes and methods prepare the data for model building, refer to the third lab tutorial in module 11 (Building a Category text predictor ). Use count_vectorizer.fit_transform().
#4 Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.
#5 Downscale the transformed data using tf-idf and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
# Extract the document term matrix
count_vectorizer = CountVectorizer(min_df=7, max_df=20)
#-----> count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(dfinal['CONTENT'])
print("\nDimensions of training data:", train_tc.shape)
vocabulary = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary)
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)
matrix_tfidf = train_tfidf.toarray()
type(matrix_tfidf)
dfeatures = pd.DataFrame(matrix_tfidf)
dfeatures['target']=dfinal['CLASS']


# 08/04/22
#6 Use pandas.sample to shuffle the dataset, set frac =1 
#7 Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s). (Do not use test_train_ split)
train_size = round(0.75*dfeatures.shape[0]) 
dfeatures_sample = dfeatures.sample(frac=1, random_state=1)
dfeatures_training = dfeatures_sample.iloc[0:train_size-1] 
dfeatures_test = dfeatures_sample.iloc[train_size:]

#8 Fit the training data into a Naive Bayes classifier.




# 09/04/22
#9  Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
#10 Test the model on the test data, print the confusion matrix and the accuracy of the model.

#1As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more happy with light skin tone emoticon.
#1Present all the results and conclusions.
#1Drop code, report and power point presentation into the project assessment folder for grading.