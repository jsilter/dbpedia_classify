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
from keras.utils import to_categorical
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
    model.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(internal_lstm_size))
    model.add(Dense(num_outputs, activation='softmax'))
    return model
    
def eval_on_dataset(dataset_path, vocab_dict, num_classes, max_input_length, steps, batch_size=100):
    start_time = datetime.datetime.now()

    _generator = create_batch_generator(dataset_path, vocab_dict, num_classes, max_input_length, batch_size)
    scores = model.evaluate_generator(_generator, steps)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('Evaluation time on %d samples: %s' % (steps*batch_size, str(elapsed_time)))
    print("Loss: %1.4f. Accuracy: %.2f%% (Chance: %0.2f%%)" % (scores[0], scores[1]*100, 100.0/num_classes))
    
    return scores, elapsed_time
    
if __name__ == "__main__":
    ## Parameters
    # Vocab Parameters
    max_vocab_size = 5000
    min_word_count = 10
    vocab_path = 'word2vec_vocab.p'
    #vocab_path = 'gensim_vocab.p'

    # Network parameters
    embedding_size = 300
    max_input_length = 500
    
    # Training parameters
    batch_size = 100
    batches_per_epoch = 100
    epochs = 20
    embedding_trainable = False
    
    loss_ = 'categorical_crossentropy'
    optimizer_ = 'adam'
    
    # Model saving parameters
    model_dir = 'models_cnn_lstm_no_train_embed_v00'
    model_path = os.path.join(model_dir, 'word2vec_cnn_lstm_{epoch:02d}.hdf5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Logging
    log_metrics = ['categorical_accuracy', 'categorical_crossentropy']
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
    
    build_own_vocab = False
    use_google_word2vec = True
    
if build_own_vocab and __name__ == "__main__":
    
        
    if not os.path.exists(vocab_path):
        
        vocab_model = Word2Vec(size=embedding_size, max_vocab_size=max_vocab_size, min_count=min_word_count, workers=2, seed=2245)
        
        print('{0}: Building own vocabulary'.format(datetime.datetime.now()))
        desc_generator = basic_desc_generator(train_path)
        vocab_model.build_vocab(desc_generator)
        print('{0}: Saving vocabulary to {1}'.format(datetime.datetime.now(), vocab_path))
        vocab_model.save(vocab_path)
    
    vocab_model = Word2Vec.load(vocab_path)
    
if use_google_word2vec and __name__ == "__main__":
    ## Google word2vec
    # Load pre-trained embeddings
    assert embedding_size == 300
    import gensim
    from gensim.models.word2vec import Word2Vec

    #Take the first bunch of words, these are sorted by decreasing count 
    #so these will be the most important, and it saves a bunch of space/time
    #Save vocab for future use
    if not os.path.exists(word2vec_model_path):
        print('Loading word2vec embeddings from {0:}'.format(google_word2vec))
        model = Word2Vec.load_word2vec_format(google_word2vec, limit=max_vocab_size, binary=True)
        model.init_sims(replace=True)
        model.save(word2vec_model_path)
    
    print('Loading saved gensim model from {0:}'.format(word2vec_model_path))
    word2vec_model = Word2Vec.load(word2vec_model_path)
    vocab_model = word2vec_model
    
if __name__ == "__main__":
    ## Main training and testing
    
    # Demo 
    # from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    # Fix random seed for reproducibility
    np.random.seed(70) # Chosen by random.org, guaranteed to be random 
    
    embedding_matrix = vocab_model.syn0
    vocab_dict = {word: vocab_model.vocab[word].index for word in vocab_model.vocab.keys()}
    vocab_size = len(vocab_dict)
    
    #Load class label dictionary
    class_ind_to_label = {}
    with open(class_labels, 'r') as cfi:
        for ind, line in enumerate(cfi):
            class_ind_to_label[ind] = line.rstrip()
    num_classes = len(class_ind_to_label)
    
    ## Create or load the model
    last_epoch, model_checkpoint_path = find_last_checkpoint(model_dir)
    initial_epoch = 0
    if model_checkpoint_path is not None:
        print('Loading epoch {0:d} from {1:s}'.format(last_epoch, model_checkpoint_path))
        model = keras.models.load_model(model_checkpoint_path)
        initial_epoch = last_epoch + 1
    else:
        print('Building new model')
        #----------------------#
        model = build_lstm_model(vocab_size, embedding_size, max_input_length, num_classes,
        embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable)
    
        model.compile(loss=loss_, optimizer=optimizer_, metrics=log_metrics)
        #-----------------------#
    
    print('Model summary')    
    print(model.summary())
    
    ## Training
    if initial_epoch < epochs:
        training_start_time = datetime.datetime.now()
        print('{0}: Starting training at epoch {1}/{2}'.format(training_start_time, initial_epoch, epochs))
        
        train_generator = create_batch_generator(train_path, vocab_dict, num_classes, max_input_length, batch_size)
        model.fit_generator(train_generator, batches_per_epoch, epochs, callbacks=_callbacks, initial_epoch=initial_epoch)
        
        training_end_time = datetime.datetime.now()
        print('{0}: Training finished at epoch {1}'.format(training_end_time, epochs))
        training_time = training_end_time - training_start_time
        print('{0} elapsed to train {1} epochs'.format(str(training_time), epochs - initial_epoch))
        
    ## Evaluation of final model
    if True:
        num_test_samples = 1000
        num_test_steps = num_test_samples // batch_size
        num_test_samples = num_test_steps * batch_size
        print('{0}: Starting testing on {1} samples'.format(datetime.datetime.now(), num_test_samples))
        test_scores, test_time = eval_on_dataset(test_path, vocab_dict, num_classes, max_input_length, num_test_steps, batch_size)
        time_per_sample = test_time.total_seconds() / num_test_samples
        print("Seconds per sample: %2.2e sec" % time_per_sample)
    
    
    