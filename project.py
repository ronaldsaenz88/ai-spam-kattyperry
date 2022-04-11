# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:08:12 2022

@author: rhuay
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.classify import NaiveBayesClassifier 
from nltk.classify.util import accuracy as nltk_accuracy 

from pre_processing import remove_non_ascii
from pre_processing import to_lowercase
from pre_processing import replace_url
from pre_processing import replace_number
from pre_processing import remove_stopwords
from pre_processing import replace_contractions
from pre_processing import remove_punct
from pre_processing import lematize_noun
from pre_processing import lematize_verb



# File path
path = "C:/Users/rhuay/Documents/CENTENNIAL/AI/PROJECT"
filename = 'Youtube02-KatyPerry.csv'
fullpath = os.path.join(path,filename)

# Read data
data = pd.read_csv(fullpath)

# --------------------------------- Initial Exploration ----------------------------------#
# remome AUTHOR, DATE, COMMENT_ID
dfinal = data.drop(columns=['AUTHOR','DATE','COMMENT_ID'])

dfinal['CONTENT2'] = dfinal['CONTENT']
dfinal['CONTENT2'] = dfinal['CONTENT2'].apply(lambda x:remove_non_ascii(x))
dfinal['CONTENT2'] = dfinal['CONTENT2'].apply(lambda x:to_lowercase(x))
dfinal['CONTENT3'] = dfinal['CONTENT2'].apply(lambda x:replace_contractions(x))
dfinal['CONTENT4'] = replace_url(dfinal['CONTENT3'])
dfinal['CONTENT5'] = replace_number(dfinal['CONTENT4'])
dfinal['CONTENT6'] = remove_punct(dfinal['CONTENT5'])
dfinal['CONTENT7'] = dfinal['CONTENT6'].apply(lambda x:lematize_noun(x))
dfinal['CONTENT8'] = dfinal['CONTENT7'].apply(lambda x:lematize_verb(x))
dfinal['CONTENT9'] = dfinal['CONTENT8'].apply(lambda x:remove_stopwords(x))