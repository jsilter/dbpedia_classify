#!/bin/bash
from __future__ import division  # Python 2 users only

import csv
import random
import datetime
import os
import re
import itertools

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

from custom_metrics import precision, recall, fmeasure, append_metric


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
    
def _brier(num_classes, preds, pick_classes):
    """
    Calculate the Brier skill score, from http://www.pnas.org/content/104/14/5959.full
    `preds` is the list of predicted probabilities, `pick_class` is the index of the class to use
    The reason we don't just take the maximum from `preds` is we might be using the `actual` value which
    might not be the predicted class
    """
    
    inv_num_cl = 1.0/num_classes
    
    denom = (1-inv_num_cl)**2 + (num_classes-1)*(inv_num_cl)**2
    numerator = 1 + K.sum(K.square(preds), axis=1)
    
    inds = tf.stack([tf.to_int64(tf.range(tf.shape(preds)[0])), pick_classes])
    t_inds = K.transpose(inds)
    sub_pick = 2*tf.gather_nd(preds, t_inds)    
    brier_pick = 1 - (numerator - sub_pick)/denom  
    
    return brier_pick
    
def brier_skill(y_true, y_pred, use_true):
    """
    Calculate Brier score, relative to either true class or predicted class
    if use_true = True, it's relative to the y_true class
    if use_true = False, it's relative to the y_pred class (how confident are we in the prediction, no knowledge of true class)
    
    """
    do_eval = False
    
    # We use this function later on for static values
    if isinstance(y_pred, np.ndarray):
        do_eval = True
        y_pred = K.variable(y_pred)
    
    num_classes = K.get_variable_shape(y_pred)[1]
    
    if use_true:
        y_pick = y_true
    else:
        y_pick = y_pred
        
    pick_classes = K.argmax(y_pick, axis=1)
    brier_out = _brier(num_classes, y_pred, pick_classes)
    
    if do_eval:
        brier_out = K.get_value(brier_out)
    
    return brier_out
    
def brier_pred(y_true, y_pred):
    return brier_skill(y_true, y_pred, False)
    
def brier_true(y_true, y_pred):
    return brier_skill(y_true, y_pred, True)
    
def make_stats(prefix, metric):
    import tensorflow as tf
    
    use_metric = metric
    if isinstance(metric, list):
        use_metric = tf.stack(metric)
      
    with tf.name_scope(prefix):
        #tf.summary.histogram(prefix, use_metric)
        tf.summary.scalar('mean', K.mean(use_metric))
        tf.summary.scalar('std', K.std(use_metric))
        tf.summary.scalar('max', K.max(use_metric))
        tf.summary.scalar('min', K.min(use_metric))
        
def make_binary_metric(metric_name, metric_func, num_classes, y_true, preds_one_hot):
    
    overall_met = [None for _ in range(num_classes)]
    with tf.name_scope(metric_name):
        for cc in range(num_classes):
            #Metrics should take 1D arrays which are 1 for positive, 0 for negative
            two_true, two_pred = y_true[:, cc], preds_one_hot[:, cc]
            cur_met = metric_func(two_true, two_pred)
            tf.summary.scalar('%d' % cc, cur_met)

            overall_met[cc] = cur_met
                
        tf.summary.histogram('overall', overall_met) 
    
def create_batch_pairwise_metrics(y_true, y_pred):
    #assert K.get_variable_shape(y_true)[1] == K.get_variable_shape(y_pred)[1]
    num_classes = K.get_variable_shape(y_pred)[1]
    preds_cats = K.argmax(y_pred, axis=1)
    preds_one_hot = K.one_hot(preds_cats, num_classes)
    
    make_binary_metric('precision', precision, num_classes, y_true, preds_one_hot)
    make_binary_metric('recall', recall, num_classes, y_true, preds_one_hot)
    make_binary_metric('fmeasure', fmeasure, num_classes, y_true, preds_one_hot)

class TensorBoardMod(keras.callbacks.TensorBoard):
    """ Modification to standard TensorBoard callback; that one
    wasn't logging all the variables I wanted """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        if self.validation_data:
            tensors = self.model.inputs + self.model.model._feed_targets
            # TODO Hard-code the unwrapping for now, not sure what's happening to make the structure so weird
            val_data = [self.validation_data[0], self.validation_data[1][0]]
            print(val_data[0].shape)
            print(val_data[1].shape)
            feed_dict = dict(zip(tensors, val_data))
            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

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
    batches_per_epoch = 10
    epochs = 100
    embedding_trainable = False
    
    loss_ = 'categorical_crossentropy'
    optimizer_ = 'adam'
    
    # Model saving parameters
    model_tag = 'cnn_lstm_no_train_embed_scratch'
    log_dir = './keras_logs_%s' % model_tag
    model_dir = 'models_%s' % model_tag
    model_path = os.path.join(model_dir, 'word2vec_%s_{epoch:02d}.hdf5' % model_tag)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    # Logging
    log_metrics = ['categorical_accuracy', 'categorical_crossentropy', brier_pred, brier_true]
    model_saver = keras.callbacks.ModelCheckpoint(model_path,verbose=1)
    tboard_saver = TensorBoardMod(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    _callbacks = [model_saver, tboard_saver]
    _#callbacks = [tboard_saver]
    
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
        _cust_objects = {'brier_skill' : brier_skill}
        model = keras.models.load_model(model_checkpoint_path, custom_objects=_cust_objects)
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
    if True:
        num_test_samples = 1000
        num_test_steps = num_test_samples // batch_size
        num_test_samples = num_test_steps * batch_size
        print('{0}: Starting testing on {1} samples'.format(datetime.datetime.now(), num_test_samples))
        test_scores, test_time = eval_on_dataset(test_path, vocab_dict, num_classes, max_input_length, num_test_steps, batch_size)
        time_per_sample = test_time.total_seconds() / num_test_samples
        print("Seconds per sample: %2.2e sec" % time_per_sample)
    
    
    