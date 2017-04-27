#!/bin/bash
from __future__ import division  # Python 2 users only

__doc__= """ Part 3 of text classification with Keras: Visualizing results"""

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

import gensim
from gensim.models.word2vec import Word2Vec

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from custom_metrics import *
from text_utils import create_batch_generator, basic_desc_generator
from utils import find_last_checkpoint
from plot_utils import plot_with_labels

if __name__ == "__main__":
    # Input parameters
    model_tag = 'cnn_lstm_fixed_embed'
    max_vocab_size = 5000
    plot_dir = 'plots'
    
    # Network parameters
    embedding_size = 300
    max_input_length = 500
    
    # Input train/test files
    train_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/train_shuf.csv'
    test_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/test_shuf.csv'
    class_labels = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/classes.txt'
    
    # Destination file for vocab
    word2vec_model_path = 'GoogleNews-vectors-negative300_top%d.model' % max_vocab_size
    
    print('Loading saved gensim model from {0:}'.format(word2vec_model_path))
    word2vec_model = Word2Vec.load(word2vec_model_path)
    vocab_model = word2vec_model
    vocab_dict = {word: vocab_model.vocab[word].index for word in vocab_model.vocab.keys()}
    
    #Load class label dictionary
    class_ind_to_label = {}
    with open(class_labels, 'r') as cfi:
        for ind, line in enumerate(cfi):
            class_ind_to_label[ind] = line.rstrip()
    num_classes = len(class_ind_to_label)
    
    batch_size = 100

if __name__ == "__main__":
    
    model_dir = 'models_%s' % model_tag
    log_dir = 'viz_logs_%s' % model_tag
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    ## Load existing model
    last_epoch, model_checkpoint_path = find_last_checkpoint(model_dir)
    initial_epoch = 0
    assert model_checkpoint_path is not None, "No model checkpoint found in %s" % model_dir
    print('Loading epoch {0:d} from {1:s}'.format(last_epoch, model_checkpoint_path))
    _cust_objects = {'brier_skill' : brier_skill, 'brier_pred': brier_pred, 'brier_true': brier_true}
    model = keras.models.load_model(model_checkpoint_path, custom_objects=_cust_objects)
    
    ## Run predictions
    if True:
        max_to_pred = 10000
        pred_res = np.zeros([max_to_pred, num_classes])
        act_res = np.zeros(max_to_pred)
        all_text = []
        all_titles = []
        print('{0}: Predicting on {1} samples'.format(datetime.datetime.now(), max_to_pred))
        pred_generator = create_batch_generator(test_path, vocab_dict, num_classes, max_input_length, batch_size, 
            return_raw_text=False, return_title=True)
        num_predded = 0
        for pred_inputs in pred_generator:
            X_pred, y_true, obj_title = pred_inputs
            #all_text += raw_text
            all_titles += obj_title
            y_preds = model.predict(X_pred)
            
            offset = num_predded
            num_predded += X_pred.shape[0]
            
            pred_res[offset:offset + y_preds.shape[0],:] = y_preds
            act_res[offset:offset + y_true.shape[0]] = np.argmax(y_true, axis=1)
            
            
            if (num_predded + batch_size) > max_to_pred:
                break
        print('{0}: Finished'.format(datetime.datetime.now()))
        
    # Plot correlation between predictions as heatmap
    pred_corr_heatmap_path = os.path.join(plot_dir, '%s_pred_correlation.png' % model_tag)
    pred_corr_hist_path = os.path.join(plot_dir, '%s_pred_histogram.png' % model_tag)
    if not os.path.exists(pred_corr_heatmap_path):
        corrs = np.corrcoef(pred_res, rowvar=0)
        heatmap_cmap = plt.get_cmap('bwr')
        
        #Histogram of correlations
        plt.figure()
        hist_indexes = np.triu_indices(corrs.shape[0])
        hist_matr = corrs[hist_indexes]
        plt.hist(corrs[:], bins=100, range=[-1.0, 1.0], alpha=1.0)
        plt.savefig(pred_corr_hist_path)
        
        plt.figure()
        heatmap = plt.pcolor(corrs, cmap=heatmap_cmap, vmin=-1.0, vmax=1.0)
        plt.title('Correlation between class predictions')
        plt.xlabel('Class Index')
        plt.ylabel('Class Label')
        locs, indexes = np.arange(num_classes, dtype=float), np.arange(num_classes, dtype=int)
        locs += 0.5
        labels = [class_ind_to_label[x] for x in indexes]
        plt.xticks(locs, indexes)
        plt.yticks(locs, labels)
        plt.colorbar()
        ax = plt.gca()
        ax.invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(pred_corr_heatmap_path)
    
    assert False
    #tSNE
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE

    # Add class centers. Have to do this before the tSNE transformation
    plot_data_points = np.concatenate([pred_res, np.identity(num_classes)], axis=0)
    plot_act_res = np.concatenate([act_res, np.arange(num_classes)])
    
    perplexity_list = [] #[5, 30, 60, 250]
    
    for perplexity in perplexity_list:
        
        tsne_vis_path = os.path.join(plot_dir, '%s_tSNE_%d_scatter.png' % (model_tag, perplexity))
        
        if os.path.exists(tsne_vis_path):
            continue
    
        tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000, random_state=2157)
        low_dim_embeds = tsne.fit_transform(plot_data_points)
        center_points = np.zeros([num_classes,2])
        
        color_map_name = 'gist_rainbow'
        cmap = plt.get_cmap(color_map_name)
        ind_to_label = class_ind_to_label
        
        plt.figure()
        plt.hold(True)
        for cc in range(num_classes):
            # Plot each class using a different color
            cfloat = (cc+1.0) / num_classes
            keep_points = np.where(plot_act_res == cc)[0]
            cur_plot = low_dim_embeds[keep_points,:]
            
            cur_color = cmap(cfloat)
            # Label the final point, that's the Probability=1 point
            peak_label = '%s_tSNE' % cc
            
            # Scatter plot
            plt.plot(cur_plot[:,0], cur_plot[:,1], 'o', color=cur_color, alpha=0.5)
            
            x, y = cur_plot[-1,:]
            plt.annotate(peak_label,
                        xy=(x, y),
                        xytext=(5, 2),
                        size='small',
                        alpha=0.8,
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
            
            
            #Plot the mean of the points, treat it as the center
            avg_label = '%d,%s' % (cc, ind_to_label[cc][0:5])
            low_dim_centers = np.mean(cur_plot, axis=0)
            low_dim_centers = low_dim_centers[np.newaxis,:]
            plot_with_labels(low_dim_centers, ['%s_Avg' % cc], color=cur_color, alpha=1.0, label=avg_label)
            
        plt.title('tSNE Visualization. Perplexity %d' % perplexity)
        plt.legend(loc='lower right', numpoints=1, fontsize=6, framealpha=0.5)
        
        plt.savefig(tsne_vis_path)
        
    # Export data to visualize with tensorboard
    # using the embedding projector
    # See https://www.tensorflow.org/get_started/embedding_viz
    from tensorflow.contrib.tensorboard.plugins import projector
    
    # Note: Must be very consistent in naming, checkpoint file has to be named the same thing as tensor (I think)
    pred_probs = tf.Variable(pred_res, name='pred_probs')
    
    metadata_path = os.path.join(log_dir, 'metadata.tsv')
    metadata_cols = ['title', 'class']
    with open(metadata_path, 'w') as met_fi:
        met_fi.write('%s\n' % '\t'.join(metadata_cols))
        for rn in range(act_res.shape[0]):
            cur_col = [all_titles[rn], '%d' % act_res[rn]]
            met_fi.write('%s\n' % '\t'.join(cur_col))
            
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = pred_probs.name
    embedding.metadata_path = metadata_path
    
    # Use the same LOG_DIR where you stored your checkpoint.
    init_op = tf.variables_initializer([pred_probs])
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver({'pred_probs': pred_probs})
        saver.save(sess, os.path.join(log_dir, "pred_probs.ckpt"))

        summary_writer = tf.summary.FileWriter(log_dir)
    
        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(summary_writer, config)
    
        #merged = tf.summary.merge_all()
        #sess.run(merged)