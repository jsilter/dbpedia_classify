#!/bin/bash
from __future__ import division  # Python 2 users only

import re

import keras
from keras import backend as K

import tensorflow as tf

class FilterTensorBoard(keras.callbacks.TensorBoard):
    """
    Write out only certain logs to a specific directory
    Intended to separate train/validation logs
    Keras adds "val_" to the beginning of all the validation metrics
    so we can include (or exclude) those
    """
    
    def __init__(self, *args, **kwargs):
        self.log_regex = kwargs.pop('log_regex', '.*')
        # Dictionary for string replacement
        self.rep_dict = kwargs.pop('rep_dict', {'val_': ''})
        super(FilterTensorBoard, self).__init__(*args, **kwargs)
        
    def filter_logs(self, logs):
        logs = logs or {}
        out_logs = {}
            
        for key in logs:
            if self.log_regex is None or re.match(self.log_regex, key):
                out_key = key
                for rep_key, rep_val in self.rep_dict.items():
                    out_key = out_key.replace(rep_key, rep_val, 1)
                out_logs[out_key] = logs[key]
        return out_logs
        
    def on_epoch_end(self, epoch, logs=None):
        super(FilterTensorBoard, self).on_epoch_end(epoch, self.filter_logs(logs))

class TensorBoardMod(keras.callbacks.TensorBoard):
    """ Modification to standard TensorBoard callback; that one
    wasn't logging all the variables I wanted """
    
    def __init__(self, *args, **kwargs):
        self.save_logs = kwargs.pop('save_logs', True)
        super(TensorBoardMod, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        if self.validation_data:
            tensors = self.model.inputs + self.model.model._feed_targets
            val_data = [self.validation_data[0], self.validation_data[1][0]]
            feed_dict = dict(zip(tensors, val_data))
            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = result[0]
            self.writer.add_summary(summary_str, epoch)
        
        if self.save_logs:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, epoch)
        self.writer.flush()