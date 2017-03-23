#!/bin/bash
from __future__ import division  # Python 2 users only

import keras
from keras import backend as K

import tensorflow as tf

class TensorBoardMod(keras.callbacks.TensorBoard):
    """ Modification to standard TensorBoard callback; that one
    wasn't logging all the variables I wanted """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        if self.validation_data:
            tensors = self.model.inputs + self.model.model._feed_targets
            val_data = [self.validation_data[0], self.validation_data[1][0]]
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