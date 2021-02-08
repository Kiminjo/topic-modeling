# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:09:05 2021

@author: Injo Kim
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
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
    
    def __init__(self, corpus):
        # fill null space to 'nan'
        self.corpus_list = corpus
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.pre_defined_stopwords = ['www', 'org', 'com', 'http', 'https', 'nbsp', 'pdf', 'license', 
                                      'licensed', 'ab ', 'ipynb', 'badge', 'shield', 'svg'
                                      '2016', '2017', '2018', '2019', '2020']
        

    def tokenize(self, corpus_list) :
        # text tokenize using nltk
        tokenized_corpus = [word_tokenize(corpus) for corpus in corpus_list]
        print('\n tokenize complete ')
        
        return tokenized_corpus
    
    def discard_puctuation(self, corpus_list) :
        # discard punctuation that pre defined
        # predefined punctuations''!()-[]{};:'"\,<>./?@#$%^&*_~''
        # capitalization of words
        remove_punctuation = [[text.lower() for text in corpus if text not in self.punctuations] for corpus in tqdm(corpus_list)]
        print('\n remove puctuation complete ')
        
        return remove_punctuation
        
    def remove_stopwords(self, corpus_list) :
        # remove stopwords
        # remove stopwords using nltk and gensim library's stopwords method
        # text that have under 2 length will be stopwords, so remove
        # 'if' can be used up to two times in one loop, so proceed in two separate ways.
        remove_stopword = [[text for text in corpus if text not in STOPWORDS if text not in stopwords.words('english')] 
                            for corpus in tqdm(corpus_list)]
        print('\n remove stopwords process1 complete ')
        
        remove_stopword = [[text for text in corpus if text not in self.pre_defined_stopwords if len(text) > 3] 
                           for corpus in tqdm(remove_stopword)]
        print('\n remove stopwords process2 complete')
        
        return remove_stopword
    
    def lemmatization(self, corpus_list) :
        # lemmatizer method call
        lemm = WordNetLemmatizer()
        
        processed_text = [[lemm.lemmatize(text) for text in corpus] for corpus in tqdm(corpus_list)]
        print('\n text preprocessing complete ')
                        
        return processed_text
    
    def text_processing(self) :
        
        tokenized = self.tokenize(self.corpus_list)
        discard_punct = self.discard_puctuation(tokenized)
        remove_stopword = self.remove_stopwords(discard_punct)
        self.result = self.lemmatization(remove_stopword)        
        
        return self.result
            

        
class GitHub_preprocessing :
    
    def __init__(self, data) : 
        
        self.data = data
        self.columns = ['owner', 'repo', 'readme', 'topics']
    
    """""""""""""""""
    data preprocessing in specific condition
    """""""""""""""""
    # repo preprocessing
    def repo_filtering(self) :
        
        self.data['readme_length'] = [len(str(text)) for text in self.data['readme']]
        self.data = self.data[(self.data['readme_length'] > 2200) & (self.data['owner_type']=='Organization')][self.necessary_columns]
            
        return self.data
    
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