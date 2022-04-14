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

import re, unicodedata

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# before you should install those libraries with pip 
# pip install contractions 
# pip install cleantext
import contractions
import cleantext
import nltk.classify.textcat as textcat


def fnc_clean_punctuation(text):
    """ Function Clean Punctuation: Remove punctuation from text """
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = textcat.TextCat.remove_punctuation(textcat, text[i])
    return new_text

def fnc_clean_contractions(text):
    """ Function Clean Contractions: Replace contrations from text """
    return contractions.fix(text)

def fnc_clean_url(text):
    """ Function Clean Contractions: Remove URLs/Links from text """
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = cleantext.replace_urls(text[i],replace_with="URL")
    return new_text

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
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = cleantext.replace_numbers(text[i],replace_with="NUM")
    return new_text    


def fnc_lemmatize_nouns(text):
    """ Function Lemmatize nouns: Used WordNetLemmatizer from ntlkt """
    lemmatizer = WordNetLemmatizer()
    new_text = []
    tokenized_words = nltk.word_tokenize(text)
    for word in tokenized_words:
        lemma = lemmatizer.lemmatize(word, pos='n')
        new_text.append(lemma)
    return " ".join(new_text)

def fnc_lemmatize_verbs(text):
    """ Function Lemmatize verbs: Used WordNetLemmatizer from ntlkt """
    lemmatizer = WordNetLemmatizer()
    new_text = []
    tokenized_words = nltk.word_tokenize(text)
    for word in tokenized_words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        new_text.append(lemma)
    return " ".join(new_text)

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
