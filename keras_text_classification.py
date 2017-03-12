#!/bin/bash
from __future__ import division  # Python 2 users only

import csv
import random
import datetime
import os

import numpy as np

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

import re


import nltk

import gensim
from gensim.models.word2vec import Word2Vec

from text_utilities import plot_with_labels

       
def build_lstm_model(top_words, embedding_vector_length, max_input_length, num_outputs,
                    internal_lstm_size=100, embedding_matrix=None, embedding_trainable=True):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_input_length, weights=[embedding_matrix], trainable=embedding_trainable))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(internal_lstm_size))
    model.add(Dense(num_outputs, activation='softmax'))
    return model
    
    
def create_training_batch(generator, num_classes, max_num=64, return_raw_text=False):
    text_data = []
    X_ = []
    y_ = []
    for info_dict in generator:
        seq_data = info_dict['int_word_list']
        _class = int(info_dict['class']) - 1
        
        text_data.append(info_dict['word_list'])
        X_.append(seq_data)
        y_.append(_class)
        
        if len(y_) >= max_num:
            break
            
    X_train = sequence.pad_sequences(X_, maxlen=max_input_length)
    y_train = np_utils.to_categorical(y_, nb_classes=num_classes)
        
    if return_raw_text:
        return X_train, y_train, text_data
    else:
        return X_train, y_train
    
    
def create_desc_generator(input_path, word2id, indefinite=False, min_word_count=10):
    columns = ['class', 'title','description']
    text_field = 'description'
    not_none = lambda x: x is not None
    _finished = False
    while not _finished:
        csv_reader = csv.DictReader(open(input_path, 'r'), fieldnames=columns)
        for ind, cur_dict in enumerate(csv_reader):
            text = cur_dict[text_field].strip().lower()
            # Don't love this but what can you do
            text = text.decode("ascii","ignore").encode("ascii")
            word_list = nltk.word_tokenize(text)
            cur_dict['word_list'] = word_list
            int_word_list = [word2id[w] for w in word_list if w in word2id]
            if len(int_word_list) < min_word_count:
                continue
            cur_dict['int_word_list'] = int_word_list
            yield cur_dict
        
        _finished = not indefinite
        
        
def create_batch_generator(input_path, word2id, num_classes, batch_size, return_raw_text=False):
    desc_generator = create_desc_generator(input_path, word2id, indefinite=True)
    while True:
        cur_batch = create_training_batch(desc_generator, num_classes, batch_size, return_raw_text=return_raw_text)
        yield cur_batch
        
        
def restore_from_checkpoint(checkpoint_dir, ptrn="*_[0-9]*.hdf5"):
    """
    Restore the most recent checkpoint in checkpoint_dir, if available.
    If no checkpoint available, does nothing.
    """
    
    import glob
    full_glob = os.path.join(checkpoint_dir, ptrn)
    all_files = glob.glob(full_glob)
    model_checkpoint_path = None
    epoch = 0
    
    for cur_fi in all_files:
        bname = os.path.basename(cur_fi)
        cur_epoch = bname.split('_')[-1].split('.')[0]
        cur_epoch = int(cur_epoch)
        if cur_epoch > epoch:
            epoch = cur_epoch
            model_checkpoint_path = cur_fi
        
    return epoch, model_checkpoint_path
    
    
def multi_to_two_class(one_hot, pos_class_num):
    """
    inarr in N x num_class, we change to N x 2
    where `pos_class_num` is the positive class and column 1,
    negative is column 0
    """
    
    #K = np
    pos_col = one_hot[:,pos_class_num]
    out_arr = K.transpose(K.stack([1-pos_col, pos_col]))
    #out_arr = pos_col
    return out_arr
    
def make_stats(prefix, metric):
    import tensorflow as tf
    
    use_metric = metric
    if isinstance(metric, list):
        use_metric = tf.pack(metric)
        
    out_dict = {'%s/mean' % prefix: K.mean(use_metric),
                '%s/std' % prefix: K.std(use_metric),
                '%s/max' % prefix: K.max(use_metric),
                '%s/min' % prefix: K.min(use_metric)}
    return out_dict
    
    
def batch_pairwise_metrics(y_true, y_pred):
    #assert K.get_variable_shape(y_true)[1] == K.get_variable_shape(y_pred)[1]
    num_classes = K.get_variable_shape(y_pred)[1]
    preds_cats = K.argmax(y_pred, axis=1)
    preds_one_hot = K.one_hot(preds_cats, num_classes)
    
    overall_precision = [None for _ in range(num_classes)]
    overall_recall = [None for _ in range(num_classes)]
    overall_fmeasure = [None for _ in range(num_classes)]
    
    out_dict = {}
    for cc in range(num_classes):
        #Metrics should take 1D arrays which are 1 for positive, 0 for negative
        two_true, two_pred = y_true[:,cc], preds_one_hot[:, cc]
        cur_dict = {
                    'precision/%02d' % cc : kmetrics.precision(two_true, two_pred),
                    'recall/%02d' % cc : kmetrics.recall(two_true, two_pred),
                    'fmeasure/%02d' % cc : kmetrics.fmeasure(two_true, two_pred),
                    'binary_accuracy/%02d' % cc: kmetrics.binary_accuracy(two_true, two_pred),
                    'act_pos/%02d' % cc: K.sum(two_true),
                    'pred_pos/%02d' % cc: K.sum(two_pred)
                    }
        out_dict.update(cur_dict)
        
        overall_precision[cc] = cur_dict['precision/%02d' % cc]
        overall_recall[cc] = cur_dict['recall/%02d' % cc]
        overall_fmeasure[cc] = cur_dict['fmeasure/%02d' % cc]
        
    out_dict.update(make_stats('precision', overall_precision))
    out_dict.update(make_stats('recall', overall_recall))
    out_dict.update(make_stats('fmeasure', overall_fmeasure))
    
    return out_dict
    
    
def briers(y_true, y_pred):
    """
    Calculate 2 brier-scores:
    'brier_true' : Brier score relative to y_true (ie real class)
    'brier_pred'   : Brier score relative to predicted class
    
    Based on the Brier skill score, from http://www.pnas.org/content/104/14/5959.full
    """
    import tensorflow as tf
    do_eval = False
    
    # We use this function later on for static values
    if isinstance(y_true, np.ndarray) and y_true is not None:
        do_eval = True
        y_true = K.variable(y_true)
    if isinstance(y_pred, np.ndarray):
        do_eval = True
        y_pred = K.variable(y_pred)
    
    num_classes = K.get_variable_shape(y_pred)[1]
    inv_num_cl = 1.0/num_classes
    
    denom = (1-inv_num_cl)**2 + (num_classes-1)*(inv_num_cl)**2
    
    # We redo the math
    numerator = 1 + K.sum(K.square(y_pred), axis=1)
    
    brier_true = None
    if y_true is not None:
        #Have to play some games to get the "true" brier score
        classes_true = K.argmax(y_true, axis=1)
        # This is a hack
        # From http://stackoverflow.com/questions/37026425/elegant-way-to-select-one-element-per-row-in-tensorflow
        inds = tf.pack([tf.to_int64(tf.range(tf.shape(y_pred)[0])), classes_true])
        t_inds = K.transpose(inds)
        sub_true = 2*tf.gather_nd(y_pred, t_inds)    
        brier_true = 1 - (numerator - sub_true)/denom   
    
    sub_pred = 2*K.max(y_pred, axis=1)
    brier_pred = 1 - (numerator - sub_pred)/denom
    
    if do_eval:
        if brier_true is not None:
            brier_true = K.get_value(brier_true)
        brier_pred = K.get_value(brier_pred)
    
    
    return brier_true, brier_pred
    
    
def briers_metrics(y_true, y_pred):
    
    brier_true, brier_pred = briers(y_true, y_pred)
    
    out_dict = {}
    out_dict.update(make_stats('brier/true', brier_true))
    out_dict.update(make_stats('brier/pred', brier_pred)) 
                
    return out_dict
    

def top_k_metric(y_true, y_pred, **kwargs):
    kk= kwargs.get('k', 3)
    return {'top_%d_cat_acc' % kk : kmetrics.top_k_categorical_accuracy(y_true, y_pred, k=kk)}
    
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
        
class BatchTimer(keras.callbacks.Callback):
        
    def on_train_begin(self, logs={}):
        self.epoch_seconds = []
        self.batch_seconds = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self._epoch_start = datetime.datetime.now()
        
    def on_batch_begin(self, batch, logs={}):
        self._batch_start = datetime.datetime.now()

    def on_batch_end(self, batch, logs={}):
        batch_time = datetime.datetime.now() - self._batch_start
        batch_seconds = batch_time.total_seconds()
        self.batch_seconds.append(batch_seconds)
        
    def on_epoch_end(self, epoch, logs={}):
        epoch_time = datetime.datetime.now() - self._epoch_start
        epoch_seconds = epoch_time.total_seconds()
        self.epoch_seconds.append(epoch_seconds)
        logs['timing/epoch_seconds'] = K.cast_to_floatx(epoch_seconds)
        logs['timing/batch_seconds/mean'] = K.cast_to_floatx(np.mean(self.batch_seconds))
        logs['timing/batch_seconds/std'] = K.cast_to_floatx(np.std(self.batch_seconds))
        
        
        
if __name__ == "__main__":
    # Parameters
    max_vocab_size = 5000
    min_word_count = 10
    
    vocab_path = 'gensim_vocab.p'
    embedding_size = 300
    max_input_length = 500
    
    samples_per_epoch = 10000
    nb_epoch = 20
    
    #log_dir = './keras_logs_cnn_lstm_scratch'
    #model_dir = 'models_cnn_lstm_scratch'
    log_dir = './keras_logs_cnn_lstm'
    model_dir = 'models_cnn_lstm'
    model_path = os.path.join(model_dir, 'word2vec_cnn_lstm_{epoch:02d}.hdf5')
    
    train_log_dir = '%s/train' % log_dir
    val_log_dir = '%s/val' % log_dir
    
    batch_size = 100
    use_google_word2vec = True
    embedding_trainable = (not use_google_word2vec) or False
    
    build_own_vocab = False
    
    train_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/train_shuf.csv'
    test_path = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/test_shuf.csv'
    class_labels = '/home/common/LargeData/TextClassificationDatasets/dbpedia_csv/classes.txt' 
    
    class_ind_to_label = {}
    with open(class_labels, 'r') as cfi:
        for ind, line in enumerate(cfi):
            class_ind_to_label[ind] = line.rstrip()
    num_classes = len(class_ind_to_label)
    
    
if True and __name__ == "__main__":
    google_word2vec = '/home/common/LargeData/GoogleNews-vectors-negative300.bin.gz'
    import gensim
    from gensim.models.word2vec import Word2Vec
    top_words = max_vocab_size
    
    word2vec_model_path = 'GoogleNews-vectors-negative300_top%d.model' % max_vocab_size

    #Take the first bunch of words, these are sorted by decreasing count 
    #so these will be the most important, and it saves a bunch of space/time
    if not os.path.exists(word2vec_model_path):
        model = Word2Vec.load_word2vec_format(google_word2vec, limit=top_words, binary=True)
        model.init_sims(replace=True)
        model.save(word2vec_model_path)
    
if use_google_word2vec and __name__ == "__main__":
    
    word2vec_model = Word2Vec.load(word2vec_model_path)
    vocab_model = word2vec_model
    embedding_matrix = vocab_model.syn0

if build_own_vocab and __name__ == "__main__":
    random.seed(10)
    vocab_model = Word2Vec(size=embedding_size, max_vocab_size=max_vocab_size, min_count=min_word_count, workers=2, seed=2245)
        
    print('{0}: Building vocabulary'.format(datetime.datetime.now()))
    vocab_model.build_vocab(desc_generator)
    print('{0}: Saving vocabulary to {1}'.format(datetime.datetime.now(), vocab_path))
    vocab_model.save(vocab_path)
    
    
if True and __name__ == "__main__":
    
    vocab_dict = {word: vocab_model.vocab[word].index for word in vocab_model.vocab.keys()}
    
    if not os.path.exists(model_dir):
        print('Creating model directory: %s' % model_dir)
        os.mkdir(model_dir)
    
    # Demo 
    # from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    # fix random seed for reproducibility
    np.random.seed(7)
    
    # Remember to add any custom objects to _cust_objects when loading
    log_metrics = ['accuracy', 'categorical_accuracy', 'categorical_crossentropy', 'kullback_leibler_divergence', top_k_metric, briers_metrics, batch_pairwise_metrics]
    #log_metrics = ['categorical_accuracy', 'categorical_crossentropy']
    
    ## create or load the model
    last_epoch, model_checkpoint_path = restore_from_checkpoint(model_dir)
    initial_epoch = 0
    if model_checkpoint_path is not None:
        print('Loading epoch {0:d} from {1:s}'.format(last_epoch, model_checkpoint_path))
        _cust_objects = {'top_k_metric': top_k_metric, 'batch_pairwise_metrics': batch_pairwise_metrics,
        'briers_metrics': briers_metrics}
        model = keras.models.load_model(model_checkpoint_path, custom_objects=_cust_objects)
        initial_epoch = last_epoch + 1
    else:
        print('Building new model')
        embedding_vector_length = embedding_size
    
        model = build_lstm_model(max_vocab_size, embedding_vector_length, max_input_length, num_classes,
        embedding_matrix=embedding_matrix, embedding_trainable=embedding_trainable)
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=log_metrics)
        
    print(model.summary())
    
    ## Training loop
    # Load models if they already exist
    model_saver = keras.callbacks.ModelCheckpoint(model_path,verbose=1)
    train_tboard_logger = FilterTensorBoard(log_dir=train_log_dir, histogram_freq=2, write_graph=True, write_images=False, log_regex=r'^(?!val).*')
    val_tboard_logger = FilterTensorBoard(log_dir=val_log_dir, histogram_freq=0, write_graph=False, write_images=False, log_regex=r"^val")

    timer = BatchTimer()
    
    _callbacks = [model_saver, timer, train_tboard_logger, val_tboard_logger]
    
    if initial_epoch < nb_epoch:
        training_start_time = datetime.datetime.now()
        print('{0}: Starting training'.format(training_start_time))
        # Lines have been randomized using unix `shuf`
        train_generator = create_batch_generator(train_path, vocab_dict, num_classes, batch_size)
        val_generator = create_batch_generator(test_path, vocab_dict, num_classes, batch_size)
        model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=_callbacks, initial_epoch=initial_epoch, validation_data=val_generator, nb_val_samples=nb_epoch)
    
    def eval_on_dataset(dataset_path, val_samples):
        start_time = datetime.datetime.now()

        _generator = create_batch_generator(dataset_path, vocab_dict, num_classes, batch_size)
        scores = model.evaluate_generator(_generator, val_samples)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print('Elapsed Time: %s' % str(elapsed_time))
        print("Loss: %1.4f. Accuracy: %.2f%%" % (scores[0], scores[1]*100))
        
        return scores, elapsed_time
        
    
    if False:
        print('{0}: Starting evaluation on entire training set'.format(datetime.datetime.now()))
        training_scores, training_time = eval_on_dataset(train_path, 560000)
        
    if True:
        num_test_samples = 1000
        print('{0}: Starting testing on {1} samples'.format(datetime.datetime.now(), num_test_samples))
        test_scores, test_time = eval_on_dataset(test_path, num_test_samples)
        time_per_sample = test_time.total_seconds() / num_test_samples
        print("Seconds per sample: %2.2e sec" % time_per_sample)
    
    if True:
       
        max_to_pred = 1000
        pred_res = np.zeros([max_to_pred, num_classes])
        act_res = np.zeros(max_to_pred)
        all_text = []
        print('{0}: Predicting on {1} samples'.format(datetime.datetime.now(), max_to_pred))
        pred_generator = create_batch_generator(test_path, vocab_dict, num_classes, batch_size, return_raw_text=True)
        num_predded = 0
        for pred_inputs in pred_generator:
            X_pred, y_true, raw_text = pred_inputs
            all_text += raw_text
            y_preds = model.predict(X_pred)
            
            brier_true, brier_pred = briers(y_true, y_preds)
            
            offset = num_predded
            num_predded += X_pred.shape[0]
            
            pred_res[offset:offset + y_preds.shape[0],:] = y_preds
            act_res[offset:offset + y_true.shape[0]] = np.argmax(y_true, axis=1)
            
            
            if (num_predded + batch_size) > max_to_pred:
                break
    
    print('{0}: Finished'.format(datetime.datetime.now()))
    
    ##Plotting
    print('{0}: Plotting'.format(datetime.datetime.now()))

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    # Correlation between predictions
    corr_preds = np.correlate
    corrs = np.corrcoef(pred_res, rowvar=0)
    heatmap_cmap = plt.get_cmap('bwr')
    
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
    plt.show()
    
    
    #tSNE
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE

    # Add class centers. Have to do this before the tSNE transformation
    plot_data_points = np.concatenate([pred_res, np.identity(num_classes)], axis=0)
    plot_act_res = np.concatenate([act_res, np.arange(num_classes)])
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=2157)
    low_dim_embeds = tsne.fit_transform(plot_data_points)
    center_points = np.zeros([num_classes,2])
    
    color_map_name = 'gist_rainbow'
    cmap = plt.get_cmap(color_map_name)
    ind_to_label = class_ind_to_label
    
    plt.figure()
    plt.hold(True)
    for cc in range(num_classes):
        cfloat = (cc+1.0) / num_classes
        keep_points = np.where(plot_act_res == cc)[0]
        cur_plot = low_dim_embeds[keep_points,:]
        point_labels = list(np.zeros_like(keep_points, dtype=str))
        point_labels[-1] = '%s_tSNE' % cc
        
        brier_true, brier_pred = briers(None, plot_data_points[keep_points,:])
        
        #Fixed color/alpha
        #cur_color = cmap(cfloat)
        #plt.scatter(cur_plot[:,0], cur_plot[:,1], color=cur_color, alpha=0.5, label=cc)
        
        #Set the alpha channel to be the Brier score
        cur_color = cmap(cfloat)
        rgba_colors = np.zeros([len(keep_points), 4])
        rgba_colors[:,:] = cur_color
        rgba_colors[:,3] = brier_pred
        plot_with_labels(cur_plot, point_labels, color=rgba_colors)
        
        label = '%d,%s' % (cc, ind_to_label[cc])
        
        #Plot the mean of the points, treat it as the center
        low_dim_centers = np.mean(cur_plot, axis=0)
        low_dim_centers = low_dim_centers[np.newaxis,:]
        #plt.scatter(low_dim_centers[0], low_dim_centers[1], s=50, color=cur_color, label=cc)
        plot_with_labels(low_dim_centers, ['%s_Avg' % cc], color=cur_color, alpha=1.0, label=label)
        
    
    plt.legend(loc='lower right')
    plt.show()
        
    
    
    