# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:09:05 2021

@author: Injo Kim
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm




class Text_preprocessing :
    
    """""""""""""""""""""
    text preprocessing proceeds in the order below.
    
    1. word tokeninze
    2. discard punctuation
    3. capitalization of words
    4. filter stop words(include short words)
    5. stemming and lemmatization
    """""""""""""""""""""
    
    def __init__(self, corpus_list):
        # fill null space to 'nan'
        self.corpus_list = corpus_list
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.url_stopwords = ['www', 'org', 'com', 'http', 'https']
        

    def tokenize(self) :
        # text tokenize using nltk
        tokenized_corpus = [word_tokenize(corpus) for corpus in self.corpus_list]
        return tokenized_corpus
    
    def text_preprocessing(self) :
        # lemmatizer method call
        lemm = WordNetLemmatizer()
        
        # get result of tokenize function
        corpus_list = self.tokenize()
        
        # * discard punctuation
        #   predefined punctuations''!()-[]{};:'"\,<>./?@#$%^&*_~''
        # * capitalization of words
        # * remove stopwords
        #   remove stopwords using nltk library's stopwords method
        #   text that have under 2 length will be stopwords, so remove
        processed_text = [[lemm.lemmatize(text.lower()) for text in corpus if text not in self.punctuations 
                          if text not in stopwords.words('english') if text not in self.url_stopwords if len(text) > 2] 
                          for corpus in tqdm(corpus_list)]
                        

        return processed_text

        



class GitHub_preprocessing :
    
    def __init__(self, data) : 
        
        self.data = data
    
    def filter(self) :
        """
        Consider only repo with 4 or more stars, forks and wath
        Not use 
        """
        event_threshold = 100
        filtered_data = self.data[(self.data['watchers_count']>event_threshold) & (self.data['stargazers_count']>event_threshold) 
                                  & (self.data['forks_count']>event_threshold)]
    
        return filtered_data
    
    def split_words(self, column_name) :
        """
        Seperate data into individual elements combined with '#'
        """
        isolated_data = [corpus.split('#') for corpus in self.data[column_name]]
    
        return isolated_data
    
    def user_organization_spliter(self, column_name) :
        """
        split user repos and organization repos
        """
        user_repo = self.data[self.data[column_name] == 'User']
        organization_repo = self.data[self.data[column_name] == 'Organization']
    
        return user_repo, organization_repo