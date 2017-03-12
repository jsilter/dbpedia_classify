#!/bin/bash
from __future__ import division  # Python 2 users only

import csv
import random
import datetime
import os
import re
import glob

def find_last_checkpoint(checkpoint_dir, ptrn="*_[0-9]*.hdf5"):
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