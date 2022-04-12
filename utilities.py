# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:36:46 2022

@author: Group 6
Ronald Saenz
Rudy Huayhua
Albert Bota

Common Functions to preprocess the data:
    
Removing punctuations like . , ! $( ) * % @
Removing URLs
Removing Stop words
Lower casing
Tokenization
Stemming
Lemmatization
"""

import re, string, unicodedata

# before you should install those libraries with pip 
# pip install contractions inflect
import contractions
import inflect

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import textcat
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def fnc_clean_punctuation(text):
    """ Function Clean Punctuation: Remove punctuation from text """
    new_text = []
    for word in text:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_text.append(new_word)
    return "".join(new_text)

def fnc_clean_contractions(text):
    """ Function Clean Contractions: Replace contrations from text """
    return contractions.fix(text)

def fnc_clean_url(text):
    """ Function Clean Contractions: Remove URLs/Links from text """
    #return re.sub(r"http\S+", "", text)
    return re.sub(r"http\S+", "URL", text)

def fnc_clean_non_ascii(text):
    """ Function Clean Non Ascii: Remove Non-Ascii characters from text """
    new_text = []
    for word in text:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_text.append(new_word)
    return "".join(new_text)

def fnc_clean_lowercase(text):
    """ Function Clean Uppercase: Replace characters to lowercase """
    new_text = []
    for word in text:
        new_word = word.lower()
        new_text.append(new_word)
    return "".join(new_text)

def fnc_clean_numbers(text):
    """ Function Clean Numbers: Replace integers characters with words from text """
    p = inflect.engine()
    new_text = []
    for word in text:
        if word.isdigit():
            new_word = p.number_to_words(word)
            #new_word = "<NUM>"
            new_word = "NUM"
            new_text.append(new_word)
        else:
            new_text.append(word)
    return "".join(new_text)

def fnc_stem_words(text):
    """ Function Stem words: Used LancasterStemmer from ntlkt """
    stemmer = LancasterStemmer()
    new_text = []
    for word in text:
        stem = stemmer.stem(word)
        new_text.append(stem)
    return "".join(new_text)

def fnc_lemmatize_nouns(text):
    """ Function Lemmatize nouns: Used WordNetLemmatizer from ntlkt """
    lemmatizer = WordNetLemmatizer()
    new_text = []
    #tokens = word_tokenize(text)
    for word in text:
        lemma = lemmatizer.lemmatize(word, pos='n')
        new_text.append(lemma)
    return "".join(new_text)

def fnc_lemmatize_verbs(text):
    """ Function Lemmatize verbs: Used WordNetLemmatizer from ntlkt """
    lemmatizer = WordNetLemmatizer()
    new_text = []
    for word in text:
        lemma = lemmatizer.lemmatize(word, pos='v')
        new_text.append(lemma)
    return "".join(new_text)

def fnc_clean_stopwords(text):
    """ Function Clean StopWords: By using stopwords from nltk, we clean the data using words in English """
    stop_words = set(stopwords.words('english'))
    tokenized_words = nltk.word_tokenize(text)
    new_text = [w for w in tokenized_words if not w.lower() in stop_words] 
    return " ".join(new_text)

def fnc_get_prediction_class(x):
    """ Function Get Prediction Class: Return the Class of the prediction value, if the value is 1, return SPAM, otherwise, return Not SPAM """
    if x == 1:
        return "Message is SPAM"
    else:
        return "Message is NOT SPAM"
