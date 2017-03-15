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
from keras.utils import to_categorical
from keras.preprocessing import sequence


def create_training_batch(generator, num_classes, max_input_length, max_batch_size=64, return_raw_text=False):
    text_data = []
    X_ = []
    y_ = []
    for info_dict in generator:
        # Pick out the sequence of integers as the words
        seq_data = info_dict['int_word_list']
        # Change the class numbering to 0-based
        _class = int(info_dict['class']) - 1
        
        text_data.append(info_dict['word_list'])
        X_.append(seq_data)
        y_.append(_class)
        
        if len(y_) >= max_batch_size:
            break
            
    # The sequences must all be the same length, so any which are shorter we pad up until the maximum input length
    X_train = sequence.pad_sequences(X_, maxlen=max_input_length)
    # Change from class number (0,1,2,etc.) to a one-hot matrix (class 0 --> [1 0 0 ..], class 2 --> [0 0 1 ...])
    y_train = to_categorical(y_, num_classes=num_classes)
        
    if return_raw_text:
        return X_train, y_train, text_data
    else:
        return X_train, y_train
        
def desc_dict_generator(input_path, fieldnames=['class', 'title','description'], text_field='description'):
    csv_reader = csv.DictReader(open(input_path, 'r'), fieldnames=fieldnames)
    for cur_dict in csv_reader:
        text = cur_dict[text_field].strip().lower()
        # Don't love this but what can you do
        text = text.decode("ascii","ignore").encode("ascii")
        cur_dict['word_list'] = nltk.word_tokenize(text)
        yield cur_dict
        
def basic_desc_generator(input_path):
    dict_generator = desc_dict_generator(input_path)
    for cur_dict in dict_generator:
        yield cur_dict['word_list']
    
def create_desc_generator(input_path, word2id, indefinite=False, min_word_count=10):
    _finished = False
    while not _finished:
        dict_generator = desc_dict_generator(input_path)
        for cur_dict in dict_generator:
            word_list = cur_dict['word_list']
            int_word_list = [word2id[w] for w in word_list if w in word2id]
            if len(int_word_list) < min_word_count:
                continue
            cur_dict['int_word_list'] = int_word_list
            yield cur_dict
        
        _finished = not indefinite
        
        
def create_batch_generator(input_path, word2id, num_classes, max_input_length, batch_size, return_raw_text=False):
    desc_generator = create_desc_generator(input_path, word2id, indefinite=True)
    while True:
        cur_batch = create_training_batch(desc_generator, num_classes, max_input_length, batch_size, return_raw_text=return_raw_text)
        yield cur_batch