#!/usr/bin/python
from __future__ import print_function

import os
import sys
import subprocess
import csv

if False and __name__ == "__main__":
    import fasttext
    model_dir = '/home/jacob/Projects/dbpedia_classify/fastText/models/'
    model_path = os.path.join(model_dir, 'dbpedia_100dim_14classes.bin')

    # Gets a memory error, bug report filed, not yet fixed
    model = fasttext.load_model(model_path, label_prefix='__label__')
    print(model)
    classifier = model

    texts = ['example very long text 1', 'example very longtext 2']

    labels = classifier.predict_proba(texts)

def line_to_dict(intoks):
    """Tokens are (class, label, class, label,...)
    In unknown class order. Parse into dict"""
    num_toks = len(intoks)
    assert num_toks % 2 == 0
    num_classes = num_toks / 2
    assert num_classes * 2 == num_toks

    out_dict = {}
    for ii in xrange(0, num_toks, 2):
        out_dict[intoks[ii]] = intoks[ii+1]
    return out_dict
        

if True and __name__ == "__main__":

    ft_exec = '/home/jacob/Software/fastText/fasttext'
    label_prefix = '__label__'
    
    if len(sys.argv) == 4:
        model_path = sys.argv[1]
        test_text_path = sys.argv[2]
        num_classes = int(sys.argv[3])
        outfile = sys.stdout
    else:
        model_dir = '/home/jacob/Projects/dbpedia_classify/fastText/models/'
        model_path = os.path.join(model_dir, 'dbpedia_100dim_14classes.bin')

        num_classes = 14
        test_text_path = '/home/jacob/Software/fastText/data/dbpedia.test'
        
        output_path = os.path.join(model_dir, 'test_results.tsv')
        outfile = open(output_path, 'w')

    predict_cmd = "{ft_exec} {cmd} {model} {test} {num_classes}".format(
        ft_exec=ft_exec, cmd='predict-prob', model=model_path, test=test_text_path,
        num_classes=num_classes)

    fieldnames = None
    out_csv = None

    proc = subprocess.Popen(predict_cmd.split(), stdout=subprocess.PIPE)

    while True:
        retcode = proc.poll()
        cur_line = proc.stdout.readline()
        if len(cur_line) == 0:
            break
        
        cur_dict = line_to_dict(cur_line.strip().split())

        assert len(cur_dict) == num_classes, 'Found %d classes but expected %d' % (len(cur_dict), num_classes)
        
        if fieldnames is None:
            
            fieldnames = sorted(cur_dict.keys(), key=lambda x: int(x.replace(label_prefix, '')))
            
            out_csv = csv.DictWriter(outfile, fieldnames, delimiter='\t')
            out_csv.writeheader()
        
        if fieldnames is not None:
            for key in fieldnames:
                assert key in cur_dict, "%s not found in line" % key
        
        out_csv.writerow(cur_dict)

    outfile.flush()
    outfile.close()
