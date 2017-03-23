#!/bin/bash
set -x

#python keras_text_classify_pt2.py with quick model_tag="cnn_lstm_fixed_embed_quick_custom"
# Default config
python keras_text_classify_pt2.py

# Allow embeddings to be trained
python keras_text_classify_pt2.py with trainable_embed

# Train vocab ourselves
python keras_text_classify_pt2.py with denovo_embed