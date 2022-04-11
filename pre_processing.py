# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:09:49 2022

@author: rhuay
"""

import re, string, unicodedata
import cleantext 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
import nltk.classify.textcat as textcat
from nltk.stem import WordNetLemmatizer



#### ------------ Pre Processing -------------------#######

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

def replace_url(text):
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = cleantext.replace_urls(text[i],replace_with="URL")
    return new_text

def replace_number(text):
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = cleantext.replace_numbers(text[i],replace_with="NUM")
    return new_text    

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    #text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(text)
    text = [w for w in tokenized_words if not w.lower() in stop_words] 
    # text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_punct(text):
    new_text = text.copy()
    for i in range(len(text) - 1):
        new_text[i] = textcat.TextCat.remove_punctuation(textcat, text[i])
    return new_text

def lematize_noun(words):
    lemmatizer = WordNetLemmatizer()
    a = []
    tokens = word_tokenize(words)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token, pos='n')
        a.append(lemmetized_word)
    return " ".join(a)


def lematize_verb(words):
    lemmatizer = WordNetLemmatizer()
    a = []
    tokens = word_tokenize(words)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token, pos='v')
        a.append(lemmetized_word)
    return " ".join(a)




