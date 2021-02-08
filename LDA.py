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

# visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ETC
from tqdm import tqdm
import warnings
import time

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
    
    parser.add_argument('--col', type=str, 
                        help='The name of the column where the corpus you want to analyze within the data is located.')
    
    parser.add_argument('--topic_number', type=int, 
                        help='Number of topics want to create')
    
    parser.add_argument('--word_number', type=int,
                        help='Number of words per topic want to create')
                                                             
    args = parser.parse_args()
    


#%%
"""""""""""""""""
data preprocessing in specific condition
"""""""""""""""""
# repo preprocessing
def repo_preprocess(df) :
    
    necessary_columns = ['owner', 'repo', 'readme', 'topics']
    
    df['readme_length'] = [len(str(text)) for text in df['readme']]
    df = df[(df['readme_length'] > 2200) & (df['owner_type']=='Organization')][necessary_columns]
        
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
    processed_result = tp.text_processing()
    
    return processed_result

# remove high frequency word 
def remove_high_frequency_word(corpus) :
    
    one_dimension_corpus = []
    predefined_high_frequency = ['learning', 'data']
    
    for word_list in corpus :
        one_dimension_corpus += word_list
    
    word_count = Counter(word_list)
    
    # Set a threshold for 98% of the total data
    threshold = np.quantile(list(word_count.values()), 0.85)
    filtered_word_count = dict(filter(lambda value : value[1]>threshold, word_count.items()))
    unwant_word = list(filtered_word_count.keys()) + predefined_high_frequency
    print(unwant_word)
    
    remove_unwant_word = [[word  for word in word_list if word not in unwant_word] for word_list in tqdm(corpus)]
    
    print('remove high frequency language is complete')
        
    return remove_unwant_word
        
    
#%% 
"""""""""""""""""
conduct LDA
get result
"""""""""""""""""
class LDA :
    def __init__(self, data, num_topics, num_words) :
        self.data = data
        self.number_of_topics = num_topics
        self.number_of_words = num_words
        self.alpha='auto'
        self.beta = 'auto'
        self.show_eval = 1
        self.passes = 20
        self.iter = 400
        
    # conduct LDA
    def lda_conduct(self) :
        
        # make word2id dictionary
        # make bag of word using word id
        self.word2id = corpora.Dictionary(self.data) 
        self.doc2bow = [self.word2id.doc2bow(corpus) for corpus in self.data]
        
        # LDA
        start = time.time()
        self.lda_model = LdaModel(self.doc2bow, num_topics=self.number_of_topics, id2word=self.word2id, passes=self.passes, iterations=self.iter,
                                   alpha=self.alpha, eta=self.beta, eval_every=self.show_eval, random_state=42)
        
        print('LDA running time : {}'.format(time.time()-start))
        
    
    # get top 20 words per topic
    def top_20_words_per_topic(self) :
        self.topics = self.lda_model.show_topics(num_words=self.number_of_words, formatted=False)
        top_20_topics = [[topic_idx, word, importance] for topic_idx, topic in self.topics for word, importance in topic]
        return top_20_topics
    
    # get topic probability per documnet 
    def topic_prob_per_doc(self) :
        topic_prob = self.lda_model.get_document_topics(self.doc2bow, per_word_topics=True)   
        return topic_prob
        
    # get topic-word matrix
    def topic_words_matrix(self) :
        matrix = self.lda_model.get_topics()
        return matrix
    
    def visualization(self) :
        
        #topics = self.lda_model.show_topics(num_words=self.number_of_words, formatted=False)
        one_dimension_data = [text for corpus in self.data for text in corpus]
        counter = Counter(one_dimension_data)
        
        out = []
        for idx, topic in self.topics:
            for word, weight in topic:
                out.append([word, idx , weight, counter[word]])
        
        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
        
        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(3, 2, figsize=(16,10), sharey=True, dpi=160)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 12000)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
            ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
        
        fig.tight_layout(w_pad=2)    
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
        plt.savefig('LDA_result/result_plot.png')
    
#%%

if __name__ == '__main__':

    # parse path
    parser()
    
    # data load
    data = pd.read_csv(args.path)
    data = remove_null_readme(repo_preprocess(data))
    processed_corpus = data[args.col]

    # text preprocessing
    processed_corpus = remove_high_frequency_word(text_preprocess(data[args.col]))

    
    # conduct LDA
    lda = LDA(data=processed_corpus, num_topics=args.topic_number, num_words=args.word_number)
    lda.lda_conduct()
    
    
    # get LDA result
    topics = lda.top_20_words_per_topic()
    prob = lda.topic_prob_per_doc() 
    topic_word_matrix = lda.topic_words_matrix()
    
    
    # save the result

    lda.visualization()
    
    pd.DataFrame(topics, columns=['topic', 'word', 'weight']).to_csv('LDA_result/top_20_topics.csv', index=False)
    pd.DataFrame(prob[12][0], columns=['topic', 'percentage']).to_csv('LDA_result/topic_prob_per_doc.csv', index=False)
    
    integer_columns = pd.DataFrame(topic_word_matrix).columns
    string_columns = [lda.word2id.id2token[element] for element in integer_columns]
    
    pd.DataFrame(topic_word_matrix, columns=string_columns, 
                 index = ['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5']).to_csv('LDA_result/topic_word_matrix.csv')
    
    print('csv save!!')
    
    


    