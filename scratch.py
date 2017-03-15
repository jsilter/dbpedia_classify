#!/bin/bash
from __future__ import division  # Python 2 users only

import csv
import random
import datetime
import os
import re

import numpy as np
import nltk

import keras
import keras.metrics as kmetrics
from keras.utils import np_utils
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K

import gensim
from gensim.models.word2vec import Word2Vec

from text_utils import create_batch_generator, basic_desc_generator
from utils import find_last_checkpoint


def build_lstm_model(top_words, embedding_size, max_input_length, num_outputs,
                    internal_lstm_size=100, embedding_matrix=None, embedding_trainable=True):
    """ 
    Parameters
    top_words : int
        Size of the vocabulary
    embedding_size : int
        Number of dimensions of the word embedding. e.g. 300 for Google word2vec
    embedding_matrix: None, or `top_words` x `embedding_size` matrix
        Initial/pre-trained embeddings
    embedding_trainable : bool
        Whether we should train the word embeddings. Must be true if no embedding matrix provided
    """
    
    if not embedding_trainable:
        assert embedding_matrix is not None, "Must provide an embedding matrix if not training one"
        
    _weights = None
    if embedding_matrix is not None:
        _weights = [embedding_matrix]
    
    model = Sequential()
    model.add(Embedding(top_words, embedding_size, input_length=max_input_length, weights=_weights, trainable=embedding_trainable))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(internal_lstm_size))
    model.add(Dense(num_outputs, activation='softmax'))
    return model
    
def eval_on_dataset(dataset_path, vocab_dict, num_classes, max_input_length, val_samples, batch_size=100):
    start_time = datetime.datetime.now()

    _generator = create_batch_generator(dataset_path, vocab_dict, num_classes, max_input_length, batch_size)
    scores = model.evaluate_generator(_generator, val_samples)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('Evaluation time on %d samples: %s' % (val_samples, str(elapsed_time)))
    print("Loss: %1.4f. Accuracy: %.2f%% (Chance: %0.2f%%)" % (scores[0], scores[1]*100, 100.0/num_classes))
    
    return scores, elapsed_time
    
if __name__ == "__main__":
    ## Parameters
    # Vocab Parameters
    max_vocab_size = 5000
    min_word_count = 10
    vocab_path = 'word2vec_vocab.p'
    vocab_path = 'gensim_vocab.p'

    # Network parameters
    embedding_size = 300
    max_input_length = 500
    
    # Training parameters
    batch_size = 100
    samples_per_epoch = 1000
    nb_epoch = 5
    embedding_trainable = False
    
    loss_ = 'categorical_crossentropy'
    optimizer_ = 'adam'
    
    # Model saving parameters
    model_dir = 'models_cnn_lstm_custom_embed_v02'
    model_path = os.path.join(model_dir, 'gensim_cnn_lstm_{epoch:02d}.hdf5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Logging
    log_metrics = ['categorical_accuracy']
    model_saver = keras.callbacks.ModelCheckpoint(model_path,verbose=1)
    _callbacks = [model_saver]
    
    # Paths to input data files
    train_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/train_shuf.csv'
    test_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/test_shuf.csv'
    class_labels = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/classes.txt' 
    
    # Input word embedding vectors
    google_word2vec = '/home/common/LargeData/GoogleNews-vectors-negative300.bin.gz'
    # Destination file for vocab
    word2vec_model_path = 'GoogleNews-vectors-negative300_top%d.model' % max_vocab_size
    
    build_own_vocab = True
    use_google_word2vec = False
    
if __name__ == "__main__":
    
    desc_generator = basic_desc_generator(train_path)
    for ind, word_list in enumerate(desc_generator):
        print(word_list[0:10])
        if ind >= 10:
            break
    
