# -*- coding: utf-8 -*-
'''
Created on Jan 16, 2017

@author: rafaelcastillo
'''
from collections import defaultdict
import glob

from keras.datasets import imdb 
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
import pandas

import cPickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split


def parse_review(path_to_file):
    '''Function to retrieve elements from review'''
    metadata = {}
    content = open(path_to_file, 'r').read()
    for element in ['author=', 'title=', 'rank=', 'maxRank=', 'source=']:
        start = content.find(element) + len(element) + 1
        end = start + content[start + 1:].find("\"") + 1
        metadata[element[:-1]] = content[start:end]
    for element in ['<body>', '<summary>']:
        start = content.find(element) + len(element)
        end = content.find(element.replace("<", "</"))
        metadata[element[1:-1]] = content[start:end]
    processed_body = parse_linguitica(path_to_file.replace('.xml','.review.pos'))
    processed_summary = parse_linguitica(path_to_file.replace('.xml','.summary.pos'))
    metadata['processed_body'] = processed_body
    metadata['processed_summary'] = processed_summary                    
    return metadata

def parse_linguitica(path_to_file, element='lexema'):
    '''Retrieves linguistic info for each word'''
    if element == 'morfemas': field = 0
    if element == 'lexema': field = 1
    text_parsed = [text.split(" ")[field] for text in open(path_to_file, 'r') if len(text) > 2]
    return " ".join(text_parsed) 


###################################################
#
# Configuration parameter:
#
###################################################

in_directory = '' # path to criticas de cine dataset
input_reviews = glob.glob(in_directory + "*.xml")

DEBUG = True  # True to print debug messages


###################################################
#
# Read files and storage results in Pandas:
#
###################################################

content = []
  
  
if DEBUG: print "Total reviews included: {0}".format(len(input_reviews))
   
for i,review in enumerate(input_reviews):
    rev_info = parse_review(review)
    if DEBUG and i<5:
        print review
        print rev_info
        print "********************************"
    content.append(rev_info)
   
df = pandas.DataFrame.from_dict(content,orient='columns')
   
if DEBUG: print "Dataframe shape: {0}".format(df.shape)
   
# Save estimator:
with open(in_directory + "parsed_reviews.pkl",'wb') as fp:
    pickle.dump(df,fp)
    
###################################################
#
# Process dataset with tfidf to get something similar to 
# the imdb dataset as here described:
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. 
#
###################################################
    
data = pickle.load(open(in_directory + "parsed_reviews.pkl", 'r'))
corpus = data['processed_summary'].tolist()
     
    
word_counts = defaultdict(int)

# Get number of occurrences by word:    
for text in corpus:
    for w in text.split(" "):
        word_counts[w.lower()] += 1
    
    
# Replace words by their occurrences:
corpus_counter = []   
for text in corpus:
    text_counts = []
    for w in text.split(" "):
        text_counts.append(word_counts[w.lower()])
    corpus_counter.append(np.array(text_counts))
       
## Add processed corpus to data dataframe:
data['corpus'] = corpus_counter
    
# Save processed summary in counts:
with open(in_directory + "parsed_reviews_processed.pkl",'wb') as fp:
    pickle.dump(data,fp) 
###################################################
#
# Excellent example from: fchollet
# 
# REPO: https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py
# Interesting Link: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
# Interesting Link: http://sebastianruder.com/word-embeddings-1/
# 
# Train a recurrent convolutional network on the IMDB sentiment
# classification task.
# 
# Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
#
###################################################
np.random.seed(1337)  # for reproducibility
 
 
# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128
 
# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4
 
# LSTM
lstm_output_size = 70
 
# Training
batch_size = 50
nb_epoch = 10
 
print('Load data and prepare it for processing...')
 
data = pickle.load(open(in_directory + "parsed_reviews_processed.pkl", 'r'))

# Simplify model's output and limit it to positive (= 1) for ranking >= 3 or negative (= 0) for ranking < 3.5:
data['rank'] = data['rank'].apply(lambda x: 0 if int(x) < 3 else x)
data['rank'] = data['rank'].apply(lambda x: 1 if int(x) >= 3 else x)

X_train, X_test, y_train, y_test = train_test_split(data['corpus'].values,data['rank'].values,test_size=0.20, stratify=data['rank'].values,random_state=400)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
  
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.05))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  
print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)




