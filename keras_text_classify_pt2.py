#!/bin/bash
from __future__ import division  # Python 2 users only

import csv
import random
import datetime
import os
import re

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

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

# Use tensorflow directly for some things
import tensorflow as tf

import gensim
from gensim.models.word2vec import Word2Vec

from text_utils import create_batch_generator, basic_desc_generator
from utils import find_last_checkpoint

from custom_metrics import *
from custom_callbacks import TensorBoardMod, FilterTensorBoard

ex = Experiment('text_classification', interactive=True)
ex.observers.append(FileStorageObserver.create('sacred_run_logs'))

@ex.capture
def build_lstm_model(vocab_size, embedding_size, max_input_length, num_outputs=None,
                    internal_lstm_size=100, embedding_matrix=None, embedding_trainable=True):
    """ 
    Parameters
    vocab_size : int
        Size of the vocabulary
    embedding_size : int
        Number of dimensions of the word embedding. e.g. 300 for Google word2vec
    embedding_matrix: None, or `vocab_size` x `embedding_size` matrix
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
    model.add(Embedding(vocab_size, embedding_size, input_length=max_input_length, weights=_weights, trainable=embedding_trainable))
    model.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(internal_lstm_size))
    model.add(Dense(num_outputs, activation='softmax'))
    return model
    
def eval_on_dataset(model, dataset_path, vocab_dict, num_classes, max_input_length, steps, batch_size=100):
    start_time = datetime.datetime.now()

    _generator = create_batch_generator(dataset_path, vocab_dict, num_classes, max_input_length, batch_size)
    scores = model.evaluate_generator(_generator, steps)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('Evaluation time on %d samples: %s' % (steps*batch_size, str(elapsed_time)))
    print("Loss: %1.4f. Accuracy: %.2f%% (Chance: %0.2f%%)" % (scores[0], scores[1]*100, 100.0/num_classes))
    
    return scores, elapsed_time

@ex.config
def default_config():
    """Default configuration. Fixed (untrainable) word embeddings loaded from word2vec"""
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
    batches_per_epoch = 10
    epochs = 100
    embedding_trainable = False
    build_own_vocab = False
    use_google_word2vec = True
    
    loss_ = 'categorical_crossentropy'
    optimizer_ = 'adam'
    
    # Model saving parameters
    model_tag = 'cnn_lstm_fixed_embed'
    
    do_final_eval = True

    # Destination file for vocab
    word2vec_model_path = 'GoogleNews-vectors-negative300_top%d.model' % max_vocab_size
    
    # Input train/test files
    train_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/train_shuf.csv'
    test_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/test_shuf.csv'
    class_labels = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/classes.txt' 
    
    # Input word embedding vectors
    google_word2vec = '/home/common/LargeData/GoogleNews-vectors-negative300.bin.gz'
    
@ex.named_config
def trainable_embed():
    """ Load Google word2vec but allow it to be trained """
    embedding_trainable = True
    build_own_vocab = False
    use_google_word2vec = True
    
    # Model saving parameters
    model_tag = 'cnn_lstm_trainable_embed'
    
@ex.named_config
def denovo_embed():
    """ Create our own vocabulary and allow it to be trained """
    embedding_trainable = True
    build_own_vocab = True
    use_google_word2vec = False
    
    # Model saving parameters
    model_tag = 'cnn_lstm_denovo_trainable_embed'
    
    
@ex.named_config
def quick():
    """ For testing, only a small number of batches"""
    # Training parameters
    batch_size = 100
    batches_per_epoch = 3
    epochs = 3
    
    embedding_trainable = True
    build_own_vocab = True
    use_google_word2vec = False
    
    # Model saving parameters
    model_tag = 'cnn_lstm_fixed_embed_quick'
    
    do_final_eval = False
    
@ex.capture
def create_vocab_model(build_own_vocab, vocab_path, embedding_size, max_vocab_size, min_word_count,
                        use_google_word2vec, word2vec_model_path, train_path, google_word2vec):
                            
    vocab_model = None
    # Build our own vocabulary from existing text
    if build_own_vocab:
    
        if not os.path.exists(vocab_path):
            
            vocab_model = Word2Vec(size=embedding_size, max_vocab_size=max_vocab_size, min_count=min_word_count, workers=2, seed=2245)
            
            print('{0}: Building own vocabulary'.format(datetime.datetime.now()))
            desc_generator = basic_desc_generator(train_path)
            vocab_model.build_vocab(desc_generator)
            print('{0}: Saving vocabulary to {1}'.format(datetime.datetime.now(), vocab_path))
            vocab_model.save(vocab_path)
        
        vocab_model = Word2Vec.load(vocab_path)
    
    if use_google_word2vec:
        ## Google word2vec
        # Load pre-trained embeddings
        assert embedding_size == 300
    
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
        
    return vocab_model
    
@ex.automain
def main_func(max_input_length, batch_size, batches_per_epoch, epochs, loss_, optimizer_, _config, do_final_eval=True):
    
    # Bring these keys into general namespace
    # Note that '_config' variable name subject to change
    string_keys = ['model_tag', 'train_path', 'test_path', 'class_labels', 'google_word2vec']
    for key in string_keys:
        exec '%s = "%s"' % (key, _config[key]) in locals()
        
    print(train_path)
    # Dynamically created logging directories
    log_dir = './keras_logs_%s' % model_tag
    train_log_dir = '%s/train' % log_dir
    val_log_dir = '%s/val' % log_dir
    custom_log_dir = '%s/custom' % log_dir
    model_dir = 'models_%s' % model_tag
    model_path = os.path.join(model_dir, 'word2vec_%s_{epoch:02d}.hdf5' % model_tag)
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Logging
    # Create callback and logging objects
    log_metrics = ['categorical_accuracy', 'categorical_crossentropy', brier_pred, brier_true]
    model_saver = keras.callbacks.ModelCheckpoint(model_path,verbose=1)
    # Log savers which play reasonably well with Keras
    train_tboard_logger = FilterTensorBoard(log_dir=train_log_dir, write_graph=False, write_images=False, log_regex=r'^(?!val).*')
    val_tboard_logger = FilterTensorBoard(log_dir=val_log_dir, write_graph=False, write_images=False, log_regex=r"^val")
    #Custom saver
    custom_tboard_saver = TensorBoardMod(log_dir=custom_log_dir, histogram_freq=0, write_graph=False, write_images=False, save_logs=False)
    _callbacks = [model_saver, train_tboard_logger, val_tboard_logger, custom_tboard_saver]
    
    # Parameters fed using Sacred
    vocab_model = create_vocab_model()
    
    ## Main training and testing
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
        _cust_objects = {'brier_skill' : brier_skill, 'brier_pred': brier_pred, 'brier_true': brier_true}
        model = keras.models.load_model(model_checkpoint_path, custom_objects=_cust_objects)
        initial_epoch = last_epoch + 1
    else:
        print('Building new model')
        #----------------------#
        model = build_lstm_model(vocab_size, num_outputs=num_classes, embedding_matrix=embedding_matrix)
        
        model.compile(loss=loss_, optimizer=optimizer_, metrics=log_metrics)
        #-----------------------#
    
    print('Model summary')    
    print(model.summary())
    
    ## Custom tensorflow logging
    # Placeholder for the true values
    y_true = model.model._feed_targets[0]
    # This is the final softmax output layer of the model
    y_pred = model.outputs[0]
    
    create_batch_pairwise_metrics(y_true, y_pred)
    
    ## Training
    if initial_epoch < epochs:
        training_start_time = datetime.datetime.now()
        print('{0}: Starting training at epoch {1}/{2}'.format(training_start_time, initial_epoch, epochs))
        
        train_generator = create_batch_generator(train_path, vocab_dict, num_classes, max_input_length, batch_size)
        
        val_size = 1000
        val_generator = create_batch_generator(test_path, vocab_dict, num_classes, max_input_length, val_size)
        val_X, val_y = val_generator.next()
        
        model.fit_generator(train_generator, batches_per_epoch, epochs, callbacks=_callbacks, initial_epoch=initial_epoch, 
            validation_data=(val_X, val_y), verbose=1)
        
        training_end_time = datetime.datetime.now()
        print('{0}: Training finished at epoch {1}'.format(training_end_time, epochs))
        training_time = training_end_time - training_start_time
        print('{0} elapsed to train {1} epochs'.format(str(training_time), epochs - initial_epoch))
        
    ## Evaluation of final model
    if do_final_eval:
        num_test_samples = 1000
        num_test_steps = num_test_samples // batch_size
        num_test_samples = num_test_steps * batch_size
        print('{0}: Starting testing on {1} samples'.format(datetime.datetime.now(), num_test_samples))
        test_scores, test_time = eval_on_dataset(model, test_path, vocab_dict, num_classes, max_input_length, num_test_steps, batch_size)
        time_per_sample = test_time.total_seconds() / num_test_samples
        print("Seconds per sample: %2.2e sec" % time_per_sample)
    