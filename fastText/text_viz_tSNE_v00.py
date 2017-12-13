#!/usr/bin/python
from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__= """ Visualize fastText classification results with tSNE"""

import sys
import csv
import random
import datetime
import os
import re
import functools
import copy

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
#from keras.preprocessing import sequence
#import keras.backend as K

import matplotlib.pyplot as plt
plt.style.use('ggplot')

os.chdir('/home/jacob/Projects/dbpedia_classify/fastText/')
import tsne_utils
from tsne_utils import calc_betas_loop, calc_betas_vec
from tsne_utils import get_squared_cross_diff_np

#sys.path.append('../')
#sys.path.append('.')
#from plot_utils import plot_with_labels

#floattype = 'float64'

#tf_float = eval('tf.%s' % floattype)
#np_float = eval('np.%s' % floattype)

#tf.contrib.keras.backend.set_floatx(floattype)

if False:
    def _make_P_ji(input, betas, in_sq_diffs=None):
        if not in_sq_diffs:
            in_sq_diffs = get_squared_cross_diff_np(input)
        tmp = in_sq_diffs * betas
        P_ji = np.exp(-1.0*tmp)
        return P_ji
        
    def _make_P_np(input, betas, batch_size):
        P_ji = _make_P_ji(input, betas)
        P_ = _get_normed_sym_np(P_ji, batch_size)
        return P_
    
    def _get_squared_cross_diff_tf(X_):
        """Compute Z_ij = ||X_(x_i) - X_(x_j)||^2 for Tensorflow Tensors"""
        batch_size = tf.shape(X_)[0]
        
        expanded = tf.expand_dims(X_, 1)
        # "tiled" is now stacked up all the samples along dimension 1
        tiled = tf.tile(expanded, tf.stack([1, batch_size, 1]))
        
        tiled_trans = tf.transpose(tiled, perm=[1,0,2])
        
        diffs = tiled - tiled_trans
        sum_act = tf.reduce_sum(tf.square(diffs), axis=2)
        
        return sum_act
        
        
    def _get_normed_sym_np(X_, batch_size=None):
        Z_ = X_
        batch_size = Z_.shape[0]
        zero_diags = 1.0 - np.identity(batch_size)
        Z_ *= zero_diags
        norm_facs = np.sum(Z_, axis=0, keepdims=True)
        Z_ = Z_ / norm_facs
        Z_ = 0.5*(Z_ + np.transpose(Z_))
        
        return Z_
        
    def _get_normed_sym_tf(X_, batch_size):
        Z_ = X_
        Z_ = tf.matrix_set_diag(Z_, tf.constant(0.0, shape=[batch_size], dtype=tf_float))
        norm_facs = tf.reduce_sum(Z_, axis=0, keep_dims=True)
        Z_ = Z_ / norm_facs
        Z_ = 0.5*(Z_ + tf.transpose(Z_))
        
        return Z_
    

if __name__ == "__main__":
    # Parametric tSNE
    
    session = tf.Session()
    # Set the session so we can use tensorflow manually later
    tf.contrib.keras.backend.set_session(session)
    
    num_input_dims = 5
    num_samps = 1600
    batch_size = 128
    # loss_ = 'mse'
    # optimizer_ = 'adam'
    
    epochs = 20
    batches_per_epoch = 8
    
    num_outputs = 2
    
    alpha_ = num_outputs - 1.0
    betas = 1.0
    
    # Generate test data
    np.random.seed(12345)
    num_clusters = 5
    cluster_centers = 5.0*np.identity(num_clusters)
    cluster_centers += cluster_centers[[1,2,3,4,0],:]
    #pick_rows = np.random.randint(num_clusters, size=num_samps)
    pick_rows = np.arange(0, num_samps) % num_clusters
    
    test_data = cluster_centers[pick_rows, :]
    test_data += np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
    #test_data = test_data.astype(np_float)
    
    tf.set_random_seed(54321)
    
    perplexity = 30
    
    import parametric_tSNE
    from parametric_tSNE import Parametric_tSNE
    
    ptSNE = Parametric_tSNE(test_data.shape[1], num_outputs, perplexity, alpha=alpha_, do_pretrain=True)
    
    ptSNE.fit(test_data, batch_size=batch_size, epochs=epochs)
    output_res = ptSNE.transform(test_data)
    
    plt.figure()
    for ci in xrange(num_clusters):
        cur_plot_rows = pick_rows == ci
        plt.plot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], 'o', label=ci, alpha=0.5)
        
    plt.legend(loc='lower right')
    plt.show()
    
    
if False:
    # Pre-train layers
    # Same architecture from 2009 paper
    all_layer_sizes = [num_input_dims, 500, 500, 2000, num_outputs]
    # hidden_layer_sizes = [500, 2000]
    all_layers = [layers.Dense(all_layer_sizes[1], input_shape=(num_input_dims,), activation='sigmoid', kernel_initializer='glorot_uniform')]
    for lsize in all_layer_sizes[2:-1]:
        cur_layer = layers.Dense(lsize, activation='sigmoid', kernel_initializer='glorot_uniform')
        all_layers.append(cur_layer)
    
    all_layers.append(layers.Dense(num_outputs, activation='linear', kernel_initializer='glorot_uniform'))
    
    X_train_tmp = test_data
    init_weights = copy.deepcopy(all_layers[0].get_weights())
    for ind, end_layer in enumerate(all_layers):
        print('Pre-training layer {0:d}'.format(ind))
        # Create AE and training
        cur_layers = all_layers[0:ind+1]
        ae = models.Sequential(cur_layers)
        
        decoder = layers.Dense(X_train_tmp.shape[1], activation='linear')
        ae.add(decoder)
        
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=10)
        
    final_weights = copy.deepcopy(all_layers[0].get_weights())
    
    #Calculate betas via perplexity
    perplexity = 30
    
    all_betas, P_matr, Hs = calc_betas_loop(test_data, perplexity)
    
    model = models.Sequential(all_layers)
    input = model.get_input_at(0)
    output = model.get_output_at(0)
    
    def _make_Q(output, batch_size):
        out_sq_diffs = _get_squared_cross_diff_tf(output)
        Q_ = tf.pow((1 + out_sq_diffs/alpha_), -(alpha_+1)/2)
        Q_ = _get_normed_sym_tf(Q_, batch_size)
        return Q_
        
    def _make_P_tf(input, betas, batch_size):
        in_sq_diffs = _get_squared_cross_diff_tf(input)
        tmp = in_sq_diffs * betas
        P_ = tf.exp(-1.0*tmp)
        P_ = _get_normed_sym_tf(P_, batch_size)
        return P_
        
    def _make_train_generator(test_data, all_betas, batch_size):
        num_steps = test_data.shape[0] // batch_size
        cur_step = -1
        while True:
            cur_step = (cur_step + 1) % num_steps
            cur_bounds = batch_size*cur_step, batch_size*(cur_step+1)
            cur_dat = test_data[cur_bounds[0]:cur_bounds[1],:]
            cur_betas = all_betas[cur_bounds[0]:cur_bounds[1]]
            
            P_array = _make_P_np(cur_dat, cur_betas, batch_size)
            
            yield cur_dat, P_array
            
    def kl_loss(y_true, y_pred, batch_size=None):
        P_ = y_true
        Q_ = _make_Q(y_pred, batch_size)
        
        _eps = tf.constant(1e-7, tf_float)
        
        kl_matr = tf.multiply(P_, tf.log(P_ + _eps) - tf.log(Q_ + _eps), name='kl_matr')
        kl_matr_keep = tf.matrix_set_diag(kl_matr, tf.constant(0.0, shape=[batch_size], dtype=tf_float))
        kl_total_cost = tf.reduce_sum(kl_matr_keep)
        
        return kl_total_cost
        
    def kl_np(P_, Q_):
        _eps = 1e-7
        kl_matr = P_ * (np.log(P_ + _eps) - np.log(Q_+_eps))
        batch_size = P_.shape[0]
        assert batch_size == Q_.shape[0]
        np.fill_diagonal(kl_matr, 0.0)
        kl_total_cost = np.sum(kl_matr)
        
        return kl_total_cost
        
    kl_loss_func = functools.partial(kl_loss, batch_size=batch_size)
    kl_loss_func.__name__ = 'KL-Divergence'
    
    model.compile('adam', kl_loss_func)
    
    feed_dict = {input: test_data}
    
    output_layer = model.layers[-1]

    output_res = model.predict(test_data)
    
    plt.figure()
    for ci in xrange(num_clusters):
        cur_plot_rows = pick_rows == ci
        plt.plot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], 'o', label=ci, alpha=0.5)
        
    plt.legend(loc='lower right')
    plt.show()


     
if False:
    input_weights = model.layers[0].get_weights()
    output_weights = model.layers[-1].get_weights()
    
    x, y = train_generator.next()

if False:
    Q_ = _make_Q(output, batch_size)
    feed_dict = {input: x}
    Q_out = Q_.eval(feed_dict=feed_dict, session=session)
    print(Q_out)
    
if False:
    # Debugging
    session = tf.Session()
    global_step = tf.Variable(0)
    init = tf.global_variables_initializer()
    session.run(init)
    
    train_generator = _make_train_generator(test_data, batch_size)
    
    num_steps = 1
    for cur_step in xrange(num_steps):
        cur_bounds = batch_size*cur_step, batch_size*(cur_step+1)
        cur_dat = test_data[cur_bounds[0]:cur_bounds[1],:]
        x, y = train_generator.next()
        
        feed_dict = {input: cur_dat}
    
        Q_ = _make_Q(output, batch_size)
        #P_out = P_.eval(feed_dict=feed_dict, session=session)
        P_out = y
        Q_out = Q_.eval(feed_dict=feed_dict, session=session)
        
        cur_kl = kl_np(P_out, Q_out)
        print(cur_kl)
        
        
        
if False:
    # For troubleshooting
    session = tf.Session()
    global_step = tf.Variable(0)
    init = tf.global_variables_initializer()
    session.run(init)
    
    # Set up training architecture and loop
    y_pred = output
    y_true = _make_P_tf(input, sigmas, batch_size)
    kl_total_cost = kl_loss(y_true, y_pred, batch_size=batch_size)
    
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(kl_total_cost, global_step=global_step)
    
    #TODO pretrain layers
    num_steps = 100
    for cur_step in xrange(num_steps):
        cur_bounds = batch_size*cur_step, batch_size*(cur_step+1)
        cur_dat = test_data[cur_bounds[0]:cur_bounds[1],:]
        cur_fd = {input: cur_dat}
        _ = session.run(train_step)
    
    
if False:
    P_out = P_.eval(feed_dict=feed_dict, session=session)
    Q_out = Q_.eval(feed_dict=feed_dict, session=session)
    
    #print(np.sum(P_out, axis=0))
    #print(np.sum(P_out, axis=1))

    
    
    
if False:
    # We have "symmetrized" the result
    mag_term = sum_act + tf.transpose(sum_act) - 2*sum_act*tf.transpose(sum_act)
    q_ = (1 + mag_term)^(-(alpha_+1)/2)
    q_ = tf.divide(q_, tf.reduce_sum(q_))
    
    """
    sum_act = sum(activations .^ 2, 2);
    Q = (1 + (bsxfun(@plus, sum_act, bsxfun(@plus, sum_act', -2 * activations * activations')) ./ v)) .^ -((v + 1) / 2);
    Q(1:n+1:end) = 0
    Q = Q ./ sum(Q(:))
    Q = max(Q, eps)
    """
    
    
if False and __name__ == "__main__":
    
    # Input train/test files
    #train_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/train_shuf.csv'
    #test_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/test_shuf.csv'
    #class_labels = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/classes.txt'
    
    dbpedia_data_dir = '/home/jacob/Software/fastText/data'
    train_path = os.path.join(dbpedia_data_dir, 'dbpedia.train')
    test_path = os.path.join(dbpedia_data_dir, 'dbpedia.test')
    class_labels = os.path.join(dbpedia_data_dir, 'classes.txt')
    
    proj_dir = '/home/jacob/Projects/dbpedia_classify/fastText'
    plot_dir = os.path.join(proj_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    
    #Load class label dictionary
    class_ind_to_label = {}
    with open(class_labels, 'r') as cfi:
        for ind, line in enumerate(cfi):
            class_ind_to_label[ind] = line.rstrip()
    num_classes = len(class_ind_to_label)
    label_prefix = '__label__'
    
    model_tag = '100dim_14classes'
    test_preds_path = os.path.join(proj_dir, 'dbpedia_100dim_14classes.test.preds.txt')
    train_preds_path = os.path.join(proj_dir, 'dbpedia_100dim_14classes.train.preds.txt')
    
    eval_path = test_path
    eval_preds_path = test_preds_path
    
    def _read_messy(in_path, delimiter=',', maxsplit=2, columns=None):
        out_data = []
        with open(in_path, 'r') as infi:
            for line in infi:
                cur_toks = line.strip().split(delimiter, maxsplit)
                out_data.append(cur_toks)
        return pd.DataFrame.from_records(out_data, columns=columns)
           
    eval_columns = ['class', 'name', 'text']
    eval_text_df = _read_messy(eval_path, columns=eval_columns)
    print(eval_text_df.head())
    
    eval_preds_df = pd.read_csv(eval_preds_path, sep='\t')
    eval_df = pd.concat([eval_text_df, eval_preds_df], axis=1)
    
    assert eval_text_df.shape[0] == eval_preds_df.shape[0]
    num_total = eval_text_df.shape[0]
    
    
    
if False:
    from timeit import default_timer as timer
    loop_times = []
    vec_times = []
    
    num_timing_iters = 10
    for iter in xrange(num_timing_iters):
        print('Iter %d/%d' % (iter, num_timing_iters))
        
        perplexity = 10 + 4*iter

        #Looped version
        start = timer()
        betas, P_matr, Hs = calc_betas_loop(test_data, perplexity)
        end = timer()
        loop_times.append(end-start)
        
        loop_betas = betas
        loop_P_matr = P_matr
        loop_Hs = Hs
        
        # Attempt the same thing vectorized
        start = timer()
        betas, P_matr, Hs = calc_betas_vec(test_data, perplexity)
        end = timer()
        vec_times.append(end-start)
        
        vec_betas = betas
        vec_P_matr = P_matr
        vec_Hs = Hs
        
        diffs = vec_betas - loop_betas
        print('Beta diffs:')
        print('%2.4e' % np.mean(np.abs(diffs)))
        
    loop_times = np.array(loop_times)
    vec_times = np.array(vec_times)
    print('Mean Loop time: %2.4f Mean Vec time: %2.4f Mean Diff: %2.4f' % (np.mean(loop_times), np.mean(vec_times), np.mean(loop_times - vec_times)))
    
if False and __name__ == "__main__":
    ## Calculate prediction accuracy
    # I think the fastText routine shuffled the training text and that was screwing me up
    # This is a simple check to make sure things are lined up right
    act_class = eval_df['class'].str.replace(label_prefix, '').astype(int).values
    pred_class = 1 + np.argmax(eval_preds_df.values, axis=1)
    
    num_correct = np.sum(act_class == pred_class)
    
    print('')
    print('%d / %d correct (%2.2f%%)' % (num_correct, num_total, 100.0*num_correct/num_total))
    
if False and __name__ == "__main__":
    
    # Plot correlation between predictions as heatmap
    pred_corr_heatmap_path = os.path.join(plot_dir, '%s_pred_correlation.png' % model_tag)
    pred_corr_hist_path = os.path.join(plot_dir, '%s_pred_histogram.png' % model_tag)
    if False and not not os.path.exists(pred_corr_heatmap_path):
    
        pred_res = eval_preds_df.values
        
        corrs = np.corrcoef(pred_res, rowvar=0)
        heatmap_cmap = plt.get_cmap('bwr')
        
        #Histogram of correlations
        plt.figure()
        hist_indexes = np.triu_indices(corrs.shape[0], 1)
        hist_matr = corrs[hist_indexes]
        plt.hist(hist_matr, bins=100, range=[-0.5, 0.5], alpha=1.0)
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
    

    #tSNE
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE
    
    num_tsne_points = 1000
    np.random.seed(789456123)
    plot_inds_ = np.random.choice(eval_df.index.values, num_tsne_points, replace=False)
    
    tsne_df = eval_df.loc[plot_inds_, :]
    class_labels = ['__label__%d'%x for x in xrange(1, num_classes+1)]
    
    pred_res = tsne_df[class_labels].values
    act_res = tsne_df['class'].values

    # Add class centers. Have to do this before the tSNE transformation
    plot_data_points = np.concatenate([pred_res, np.identity(num_classes)], axis=0)
    plot_act_res = np.concatenate([act_res, np.arange(num_classes)])
    
    perplexity_list = [5, 30, 60, 250]
    
    for perplexity in perplexity_list:
        
        tsne_vis_path = os.path.join(plot_dir, '%s_tSNE_%d_scatter.png' % (model_tag, perplexity))
        
        if False and os.path.exists(tsne_vis_path):
            continue
    
        print('Beginning perplexity %d' % perplexity)
        
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
                        alpha=0.6,
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
            
            
            #Plot the mean of the points, treat it as the center
            avg_label = '%d,%s' % (cc, ind_to_label[cc][0:5])
            low_dim_centers = np.mean(cur_plot, axis=0)
            low_dim_centers = low_dim_centers[np.newaxis,:]
            #plot_with_labels(low_dim_centers, ['%s_Avg' % cc], color=cur_color, alpha=1.0, label=avg_label)
            
        plt.title('tSNE Visualization. Perplexity %d' % perplexity)
        plt.legend(loc='lower right', numpoints=1, fontsize=6, framealpha=0.5)
        
        plt.savefig(tsne_vis_path)
 
if False and __name__ == "__main__":
    
    ## Run predictions
    if True:
        max_to_pred = 10000
        pred_res = np.zeros([max_to_pred, num_classes])
        act_res = np.zeros(max_to_pred)
        all_text = []
        all_titles = []
        print('{0}: Predicting on {1} samples'.format(datetime.datetime.now(), max_to_pred))
        pred_generator = create_batch_generator(eval_path, vocab_dict, num_classes, max_input_length, batch_size, 
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