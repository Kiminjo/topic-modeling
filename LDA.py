# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 01:01:29 2021

@author: injo Kim

Developer recommendation using GitHub data 
LDA conduct on readme data
"""

# data analysis
import numpy as np
import pandas as pd
from collections import Counter

# text preprocessing
from preprocessing import Text_preprocessing as t_p

# LDA 
from gensim import corpora
from gensim.models import LdaModel

# parser
import argparse


# ETC
from tqdm import tqdm
from pprint import pprint
import warnings

warnings.filterwarnings(action='ignore')

#%%
"""""""""""""""""
Define parser in command line
"""""""""""""""""

def parser() :
    
    global args
    parser = argparse.ArgumentParser()
    
    # add argument about path 
    parser.add_argument('--path', type=argparse.FileType('r', encoding='UTF-8'), 
                        help='Directory where the input data exists.')
    
    parser.add_argument('--owner', type=str,
                        help='owner of corpus want to conduct LDA')
    
    parser.add_argument('--mode', type=str, 
                        help='["save", "print"] If save mode, save result to a csv file. else print mode, print result to the command prompt')
    
    args = parser.parse_args()
    


#%%
"""""""""""""""""
data preprocessing in specific condition
"""""""""""""""""
# repo preprocessing
def repo_preprocess(df) :
    
    necessary_columns = ['owner', 'repo', 'readme', 'topics']
    df = df[(df['owner_type']=='Organization') & (df['contributors_count'] > 5)][necessary_columns]
    
    return df

# remove null readme rows 
def remove_null_readme(df) :
    
    df = df.dropna(axis=0, subset=['readme'])
    df = df.reset_index(drop=True)
    
    return df

#%%
"""""""""""""""""
text preprocessing follows below steps.

 1. word tokeninze
 2. discard punctuation
 3. capitalization of words
 4. filter stop words(include short words)
 5. stemming and lemmatization
"""""""""""""""""

# text preprocessing
def text_preprocess(corpus) :
    
    tp = t_p(corpus)
    processed_result = tp.text_preprocessing()
    
    return processed_result

# remove high frequency word 
def remove_high_frequency_word(corpus) :
    
    result = []
    
    for word_list in corpus :
        word_count = Counter(word_list)
        
        # Set a threshold for 98% of the total data
        threshold = np.quantile(list(word_count.values()), 0.98)
        filtered_word_count = dict(filter(lambda value : value[1]>threshold, word_count.items()))
        unwant_word = list(filtered_word_count.keys())
        
        remove_unwant_word = [word for word in word_list if word not in unwant_word]
        result.append(remove_unwant_word)
        
    return result
        
    
#%% 
"""""""""""""""""
conduct LDA
get result
"""""""""""""""""

# conduct LDA
def lda_conduct(word_list) :
    
    # make word2id dictionary
    # make bag of word using word id
    word2id = corpora.Dictionary(word_list) 
    d2b = [word2id.doc2bow(corpus) for corpus in word_list]
    
    # LDA
    lda_model = [LdaModel([corpus], num_topics=5, id2word=word2id, passes=30) for corpus in tqdm(d2b)]
    
    return word2id, d2b, lda_model

# get top 20 words per topic
def top_20_words_per_topic(model) :
    topics = model.show_topics(num_words=20, formatted=False)
    top_20_topics = [[topic_idx, word, importance] for topic_idx, topic in topics for word, importance in topic]
    return top_20_topics

# get topic probability per documnet 
def topic_prob_per_doc(doc2bow, model) :
    topic_prob = model.get_document_topics(doc2bow, per_word_topics=True)   
    return topic_prob
    
# get topic-word matrix
def topic_words_matrix(model) :
    matrix = model.get_topics()
    return matrix
    
#%%

if __name__ == '__main__':

    # data load
    data = pd.read_csv('..\..\data\github.csv')
    data = remove_null_readme(repo_preprocess(data))
    #processed_corpus = data[data['owner']==args.owner]['readme']
    processed_corpus = data['readme']

    # text preprocessing
    processed_corpus = remove_high_frequency_word(text_preprocess(processed_corpus))
    
    # conduct LDA
    id2word, doc2bow, lda_model = lda_conduct(processed_corpus)
    
    topics = [top_20_words_per_topic(model) for model in lda_model]
    prob = [topic_prob_per_doc(d2b, lda_model[idx]) for idx, d2b in enumerate(doc2bow)]
    topic_word_matrix = [topic_words_matrix(model) for model in lda_model]

"""    
    if args.mode == 'print' :
        print('get top 20 words per topic \n')
        pprint(topics)
        print('======================================================== \n')
        print('get topic probability per documnet \n')
        pprint(prob)
        print('======================================================== \n')
        print('get topic-word matrix \n')
        pprint(topic_word_matrix)
        
    elif args.mode == 'save' :
        pd.DataFrame(topics[0], columns=['topic', 'word', 'weight']).to_csv('LDA_result/top_20_topics_' + args.owner +'.csv', index=False)
        pd.DataFrame(prob[0][0]).to_csv('LDA_result/topic_prob_per_doc_' + args.owner +'.csv', index=False)
        
        integer_columns = pd.DataFrame(topic_word_matrix[0]).columns
        string_columns = [id2word.id2token[element] for element in integer_columns]
        
        pd.DataFrame(topic_word_matrix[0], columns=string_columns, 
                     index = ['topic0', 'topic1', 'topic2', 'topic3', 'topic4']).to_csv('LDA_result/topic_word_matrix_'+ args.owner +'.csv')
        
        print('csv save!!')

"""
    