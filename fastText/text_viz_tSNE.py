#!/usr/bin/python
from __future__ import print_function
from __future__ import division

import os
import sys
import subprocess
import csv
import datetime
import argparse

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_pdf import PdfPages

import parametric_tSNE
from parametric_tSNE import Parametric_tSNE

DEFAULT_EPS = 1e-7

def _plot_kde(output_res, act_classes, color_palette, alpha=0.5, center_labels=None, plot_classes=None):
    uniq_classes = set(act_classes)
    uniq_classes = list(sorted(uniq_classes))
    if plot_classes is None:
        num_clusters = len(uniq_classes)
        plot_classes = list(uniq_classes)
    if center_labels == True:
        center_labels = ['%s' % x for x in plot_classes]
    if center_labels is not None:
        assert len(center_labels) == num_clusters, 'If center_labels provided they must be the same length as the number of clusters'
            
    for pc in plot_classes:
        cur_plot_rows = act_classes == pc
        color_ind = uniq_classes.index(pc)
        cur_cmap = sns.light_palette(color_palette[color_ind], as_cmap=True)
        sns.kdeplot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], cmap=cur_cmap, shade=True, alpha=alpha, shade_lowest=False)
        
        center_x = np.median(output_res[cur_plot_rows, 0])
        center_y = np.median(output_res[cur_plot_rows, 1])
        
        if center_labels is not None:
            plt.text(center_x, center_y, center_labels[pc], horizontalalignment='center')


def _plot_scatter(output_res, act_classes, alpha=0.5, symbol='o', plot_classes=None):
    uniq_classes = set(act_classes)
    uniq_classes = list(sorted(uniq_classes))
    if plot_classes is None:
        num_clusters = len(uniq_classes)
        plot_classes = list(uniq_classes)
    for pc in plot_classes:
        cur_plot_rows = act_classes == pc
        color_ind = uniq_classes.index(pc)
        cur_color = color_palette[color_ind]
        cur_color = map(float, cur_color)
        cur_color = matplotlib.colors.to_rgba(cur_color)
        plt.plot(output_res[cur_plot_rows, 0], output_res[cur_plot_rows, 1], symbol, color=cur_color, label=pc, markersize=0.1)


def load_class_label_dict(class_labels_path):
    #Load class label dictionary
    # First column should be classes. Second column can be indexes
    # If indexes not provided, just use continuous row numbers
    class_ind_to_label = {}
    class_label_to_ind = {}
    with open(class_labels_path, 'r') as cfi:
        for ind, line in enumerate(cfi):
            cur_line = line.rstrip()
            cur_toks = cur_line.split('\t')
            cur_ind = ind
            cur_label = cur_toks[0]
            if len(cur_toks) >= 2:
                cur_ind = int(cur_toks[1])
            class_ind_to_label[cur_ind] = cur_label
            class_label_to_ind[cur_label] = cur_ind
    return class_ind_to_label, class_label_to_ind


def _argparser():
    parser = argparse.ArgumentParser(description="Visualize tSNE projection of text processed with fastText")
    parser.add_argument('prob_path', type=str, help="""Path to predicted probabilities for each class. Should be in a neat matrix form, with column labels of form <label_prefix><class number> (designed for dbPedia)""")
    parser.add_argument('known_labels', type=str, help="""Test text file. Only important element is having class labels in the first column""")
    parser.add_argument('--outfigurepath', type=str, default=None, help="""PDF path into which to save tSNE (and other) plots""")
    parser.add_argument('--num_training_samples', type=int, default=2000, help="""Number of samples from which to train the model. Default %(default)s""")
    parser.add_argument('--classes_path', default=None, type=str, required=True, help="""Text file listing class names""")
    parser.add_argument('--label_prefix', default=None, type=str, required=True, help="""Prefix to remove from class labels""")
    
    parser.add_argument('--excl_class', default=None, type=int, required=False, help="""Of the class label indexes, which do not exist in training set. The plots are altered somewhat if this is set.""")

    parser.add_argument('--model_path', default=None, type=str, help="""Path to which to load/save trained model""")
    parser.add_argument('--override', default=False, action='store_true', help="""Re-train model even if `model_path` exists. Will overwrite that model""")
    parser.add_argument('--debug', default=False, action='store_true', help="""Debug mode. Reduces training parameters to be fast and inaccurate""")
    parser.add_argument('--seed', default=0, type=int, help="""Seed for random state""")

    return parser

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        args = _argparser().parse_args()
        prob_path = args.prob_path
        known_labels = args.known_labels
        outfigurepath = args.outfigurepath
        num_training_samples = args.num_training_samples
        classes_path = args.classes_path
        label_prefix = args.label_prefix
        model_path = args.model_path
        override = args.override
        debug = args.debug
        excl_class = args.excl_class
        seed = args.seed
    else:
        label_prefix = '__label__'

        # Path to the test text
        # Only important element is having the class labels in the first column
        known_labels = '/home/jacob/Software/fastText/data/dbpedia.test'
        # Class labels
        classes_path = '/home/jacob/Software/fastText/data/dbpedia_csv/classes.txt'
        #model_dir = '/home/jacob/Projects/dbpedia_classify/fastText'
        prob_path = 'dbpedia_100dim_14classes.test.preds.txt'
        model_path = 'dbpedia_tSNE.h5'
        override = False
        num_training_samples = 2000
        debug = True
        outfigurepath = 'dbpedia_viz_scratch.pdf'
        seed = 0
        excl_class = 14

    perplexity = None
    do_pretrain = True
    num_outputs = 2
    alpha_ = num_outputs - 1
    
    batch_size = 128
    epochs = 12
    
    out_pdf = None
    if outfigurepath:
        out_pdf = PdfPages(outfigurepath)
    
    all_labels = []
    with open(known_labels, 'r') as ttp:
        for line in ttp:
            toks = line.split(',', 2)
            all_labels.append(toks[0])
    all_labels = np.array(all_labels)

    class_ind_to_label, class_label_to_ind = load_class_label_dict(classes_path)
    num_classes = len(class_ind_to_label)
    all_labels_int = np.char.replace(all_labels, label_prefix, '').astype(int)
    
    all_prob_data_df = pd.read_csv(prob_path, sep='\t')

    if debug:
        model_path = 'dbpedia_tSNE_debug.h5'
        do_pretrain = False
        num_training_samples = 500
        max_debug_pts = 10000
        all_prob_data_df = all_prob_data_df.iloc[0:max_debug_pts, :]
        all_labels_int = all_labels_int[0:max_debug_pts]
        
    # Reorder training data columns
    col_classes_int = all_prob_data_df.columns.str.replace(label_prefix, '').astype(int).values
    new_order = np.argsort(col_classes_int)
    all_prob_data_df = all_prob_data_df.ix[:, new_order]
    
    # Add prior to shrink clusters closer
    # Add noise to training data to prevent data from being clustered too closely together
    # Mostly this is to make visualization a little nicer to look at. In the clustering sense
    # it decreases performance
    np.random.seed(12345)
    pseudo_prior = 1.0/num_classes
    all_prob_data_df += np.random.uniform(pseudo_prior, 1.0*pseudo_prior, size=all_prob_data_df.shape)
    # Renormalize
    all_prob_data_df /= np.ones_like(all_prob_data_df)*all_prob_data_df.values.sum(axis=1, keepdims=True)
    
    # Isolate initial training data used to created neural network
    training_prob_data = all_prob_data_df.values[0:num_training_samples , :]
    # training_prob_classes = all_labels[0:num_training_samples]
    training_classes_int = all_labels_int[0:num_training_samples]
    
    # Since our input data are probabilities, we can just set betas to 1
    training_betas = np.ones(training_prob_data.shape[0], dtype=float)
    
    # For timing comparison, use scikit-learn to train tSNE model
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE
    print('{time}: Transforming {points} points using scikit-learn tSNE'.format(time=datetime.datetime.now(), points=training_prob_data.shape[0]))
    # Just use a fixed perplexity of 30 here for simplicity
    sk_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=seed)
    sk_low_dim_embeds = sk_tsne.fit_transform(training_prob_data)
    print('{time}: Finished Transforming {points} points using scikit-learn tSNE'.format(time=datetime.datetime.now(), points=training_prob_data.shape[0]))
    
    ## Train tSNE model
    ptSNE = Parametric_tSNE(training_prob_data.shape[1], num_outputs, perplexity, alpha=alpha_, do_pretrain=do_pretrain, batch_size=batch_size, seed=seed)
    
    if override or not os.path.exists(model_path):
        print('{time}: Training model {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
        ptSNE.fit(training_prob_data, training_betas=training_betas, epochs=epochs)
        print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
        ptSNE.save_model(model_path)
    else:
        print('{time}: Loading from {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
        ptSNE.restore_model(model_path)
    
    transformed_training_res = ptSNE.transform(training_prob_data)
    
    rest_prob_data = all_prob_data_df.values[num_training_samples:, :]
    rest_classes_int = all_labels_int[num_training_samples:]
    
    transformer_list = [{'tag': 'tSNE', 'transformer': ptSNE}]
    plot_pca = False
    if plot_pca:
        from sklearn.decomposition import PCA
        pca_transformer = PCA(n_components=2)
        pca_transformer.fit(training_prob_data)
        transformer_list.append({'tag': 'PCA', 'transformer': pca_transformer})
        
    print('{time}: Transforming other {points} points'.format(time=datetime.datetime.now(), points=rest_prob_data.shape[0]))
    transformed_rest = ptSNE.transform(rest_prob_data)
    print('{time}: Transformation of {points} points complete'.format(time=datetime.datetime.now(), points=rest_prob_data.shape[0]))
    
    for transformer_dict in transformer_list:
        transformer = transformer_dict['transformer']
        tag = transformer_dict['tag']
        
        print('{time}: {tag} Transforming other {points} points'.format(time=datetime.datetime.now(), points=rest_prob_data.shape[0], tag=tag))
        
        transformed_training_res = transformer.transform(training_prob_data)
        transformed_rest = transformer.transform(rest_prob_data)
        
        # Plot training data
        color_palette = sns.color_palette("hls", num_classes)
        _plot_kde(transformed_training_res, training_classes_int, color_palette, center_labels=class_ind_to_label)
        
        # Plot the remaining data as a scatter plot
        plot_classes = sorted(set(rest_classes_int))
        if excl_class:
            plot_classes = [int(excl_class)]
            
        _plot_scatter(transformed_rest, rest_classes_int, color_palette, plot_classes=plot_classes)
        
        title_lines = ['%s Results' % tag]
        if excl_class:
            title_lines.append('%s classes; excluded %s' % (num_classes - 1, class_ind_to_label[excl_class]))
        else:
            title_lines.append('All %d classes' % num_classes)
        title_str = '\n'.join(title_lines)
        plt.title(title_str)
        
        # Plot legend only if we didn't label the classes ourselves
        if class_ind_to_label is None:
            leg = plt.legend(bbox_to_anchor=(1.0, 1.0))
            # Set marker to be fully opaque in legend
            for lh in leg.legendHandles: 
                lh._legmarker.set_alpha(1.0)
                lh._legmarker.set_markersize(5.0)
    
        # Hide axes ticks
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticklabels([])
        cur_axes.axes.get_yaxis().set_ticklabels([])
    
        if out_pdf:
            plt.savefig(out_pdf, format='pdf')
        
        
    # Joy plot
    # Initialize the FacetGrid object
    if excl_class:
        joy_class = int(excl_class)
        joy_class_df = all_prob_data_df.loc[all_labels_int == joy_class, :]
        melted_df = joy_class_df.melt(var_name='class', value_name='value')
        melted_df['class'] = melted_df['class'].str.replace(label_prefix, '')
        melted_df['class'] = melted_df['class'].astype(int)
        melted_df['class_label'] = np.array([class_ind_to_label[x] for x in melted_df['class'].values])
        melted_df['log10val'] = np.log10(melted_df['value'] + DEFAULT_EPS)
        
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        
        #val_key = 'value'
        #xlims = (0.0, 1.0)
        
        val_key = 'log10val'
        xlims = (np.log10(pseudo_prior/10.0), -0.1)
        
        g = sns.FacetGrid(melted_df, row="class_label", hue="class_label", aspect=15, size=10.0/num_classes, palette=pal, xlim=xlims)
        
        # Draw the densities in a few steps
        g.map(sns.kdeplot, val_key, clip_on=False, shade=True, lw=1, alpha=1, bw=.1)
        g.map(sns.kdeplot, val_key, clip_on=False, color="w", lw=1, bw=.1)
        #g.map(plt.axhline, y=0, lw=2, clip_on=False)
        
        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color, 
                    ha="right", va="center", transform=ax.transAxes)
        
        g.map(label, "class_label")
        
        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.05)
        g.fig.subplots_adjust(left=0.15, right=1.0)
        
        # Remove axes details that don't play will with overlap
        g.set_titles("")
        g.set(yticklabels=[])
        g.despine(bottom=True, left=True)
        
        plt.xlabel('Probability')
        locs, labels = plt.xticks()
        new_labels = []
        for label in labels:
            old_label = label.get_text()
            new_label = old_label
            if old_label:
                new_label = new_label.replace(u'\u2212', '-') #Seriously Python?
                new_label = '{0:0.2f}'.format(10.0**float(new_label))
            new_labels.append(new_label)
        g.set(xticklabels=new_labels)
        
        g.fig.suptitle('Distribution of predictions for %s' % (class_ind_to_label[excl_class]))
        
        if out_pdf:
            plt.savefig(out_pdf, format='pdf')
    
    if out_pdf:
        out_pdf.close()
        
    if not out_pdf:
        plt.show()
