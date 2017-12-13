#!/usr/bin/python
from __future__ import print_function
from __future__ import division

__doc__ == """ Script for inspecting individual rows of interest in dbPedia database
and the classification results thereof"""

import os
import sys
import subprocess
import csv
import datetime
import argparse

import numpy as np
import pandas as pd

from text_viz_tSNE import load_class_label_dict

if __name__ == "__main__":
    
    prob_file_name = 'dbpedia_100dim_all_14_classes.test.preds.txt'
    prob_dir = '/home/jacob/Projects/dbpedia_classify/fastText/dbpedia_all_14_classes'
    prob_path = os.path.join(prob_dir, prob_file_name)
    
    orig_text_path = os.path.join(prob_dir, 'dbpedia.test')
    label_prefix = '__label__'
    classes_path = os.path.join(prob_dir, 'classes.txt')
    
    class_ind_to_label, class_label_to_ind = load_class_label_dict(classes_path)
    num_classes = len(class_ind_to_label)
    
    # We do our own parsing because the text isn't quoted
    # and pandas chokes when there is text with the delimiter
    names=['label', 'name', 'text']
    
    all_labels = []
    all_text_dicts = []
    with open(orig_text_path, 'r') as ttp:
        for line in ttp:
            toks = line.split(',', 2)
            all_labels.append(toks[0])
            class_ind = int(toks[0].replace(label_prefix, ''))
            class_name = class_ind_to_label[class_ind]
            cur_dict = {'label': toks[0], 'class_ind': class_ind, 
                'class_name': class_name, 'name': toks[1], 'text': toks[2]}
            all_text_dicts.append(cur_dict)
        
    orig_text_df = pd.DataFrame.from_dict(all_text_dicts)
    del all_text_dicts
    
    all_prob_data_df = pd.read_csv(prob_path, sep='\t')
    label_col_names = all_prob_data_df.columns
    
    # In theory these are in the same order
    all_data_df = pd.concat([orig_text_df, all_prob_data_df], axis=1, ignore_index=False)
    
    # Sort just for consistency
    all_data_df.sort_values(by=['class_ind', 'name'], inplace=True)
    
    selections = [
        {'filter_class' : 'Plant', 'target_vector': {'Plant': 0.5, 'Animal': 0.5}, 'desc': 'Halfway between Plant/Animal'},
        {'filter_class' : 'Animal', 'target_vector': {'Plant': 0.5, 'Animal': 0.5}, 'desc': 'Halfway between Plant/Animal'},
        {'filter_class' : 'Building', 'target_vector': {'Company': 1.0}, 'desc': 'Closest to Company'},
        {'filter_class' : 'Company', 'target_vector': {'Building': 1.0}, 'desc': 'Closest to Building'},
        {'filter_class' : 'EducationalInstitution', 'target_vector': {'Company': 1.0}, 'desc': 'Closest to Company'},
        {'filter_class' : 'Building', 'target_vector': {'EducationalInstitution': 1.0}, 'desc': 'Closest to EducationalInstitution'},
        {'filter_class' : 'OfficeHolder', 'target_vector': {'Artist': 1.0}, 'desc': 'Closest to Artist'},
        {'filter_class' : 'OfficeHolder', 'target_vector': {'Athlete': 1.0}, 'desc': 'Closest to Athlete'}
    ]
    
    for selection in selections:
        fc = selection['filter_class']
        cur_df = all_data_df.loc[all_data_df['class_name'] == fc, :]
        tvec = np.zeros([1, num_classes])
        
        for cname, cval in selection['target_vector'].iteritems():
            tvec[0, class_label_to_ind[cname]-1] = cval
        
        diffs = cur_df[label_col_names].values - tvec
        sq_diffs = np.sqrt(np.sum(diffs**2, axis=1))
        closest_ind = np.argmin(sq_diffs)
        dist = sq_diffs[closest_ind]
        
        selection['result'] = cur_df.iloc[closest_ind]
        selection['result_dist'] = dist
        
    # Print results
    outformat= ['FilterClass', 'FilterDescription', 'Name', 'Text', 'Distance', 'ClassProb']
    print('\t'.join(outformat))
    for selection in selections:
        result = selection['result']
        fill_dict = {'fc': selection['filter_class'], 'desc': selection['desc'], 'name': result['name'], 'text': result['text'], 'dist': selection['result_dist']}
        fill_dict['class_prob'] = result['__label__{0:d}'.format(result['class_ind'])]
        outline = '{fc} {desc}   {name}  {text} {dist} {class_prob}'.format(**fill_dict)
        print(outline)