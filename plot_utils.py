#!/usr/bin/env python2.7
__doc__ = """Utility functions for processing text"""

import itertools
import gzip
import json

import nltk
import numpy as np

import scipy
import scipy.stats
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
plt.style.use('ggplot')

    
def plot_with_labels(low_dim_embs, labels, filename=None, text_alpha=0.8, **plot_kwargs):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #plt.figure()  #in inches
    
    xx, yy = low_dim_embs[:,0], low_dim_embs[:,1]
    plt.scatter(xx, yy, **plot_kwargs)
    for ii, label in enumerate(labels):
        x, y = xx[ii], yy[ii]
        try:
            plt.annotate(label,
                        xy=(x, y),
                        xytext=(5, 2),
                        size='small',
                        alpha=text_alpha,
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        except Exception as ex:
            pass
            #raise ex
                        
    if filename is not None:
        plt.title(filename.split('.')[0])
        plt.savefig(filename)
                        
def tsne_plot(data_points, filename=None):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=2157)
    low_dim_embs = tsne.fit_transform(data_points)
    plt.scatter(low_dim_embeds[:,0], low_dim_embeds[:,1])
        
    if filename is not None:
        plt.title(filename.split('.')[0])
        plt.savefig(filename)