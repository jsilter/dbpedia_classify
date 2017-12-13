#!/bin/bash

excl_class=${1:-"14"}

label_prefix=__label__
class_key=${label_prefix}${excl_class}

orig_data_dir=dbpedia
new_data_dir=dbpedia_excl_${excl_class}

mkdir -p ${new_data_dir}

# Create training/test set which exclude one of the classes
# Note the trailing space so that __label__1 doesn't exclude __label__10
grep --invert-match "^${class_key} " ${orig_data_dir}/dbpedia.train > ${new_data_dir}/dbpedia_excl_${excl_class}.train

grep --invert-match "^${class_key} " ${orig_data_dir}/dbpedia.test > ${new_data_dir}/dbpedia_excl_${excl_class}.test

cp ${orig_data_dir}/*.txt ${new_data_dir}/

model_dir="${new_data_dir}/models"
dbpedia_data_dir=${new_data_dir}
num_classes=13

model_label="dbpedia_100dim_excl_${excl_class}"
train_set="${dbpedia_data_dir}/dbpedia_excl_${excl_class}.train"
test_set="${dbpedia_data_dir}/dbpedia_excl_${excl_class}.test"
orig_test_set=${orig_data_dir}/dbpedia.test

output=${model_dir}/${model_label}
output_bin="${output}.bin"
train_preds="${new_data_dir}/${model_label}.train.preds.txt"
test_preds="${new_data_dir}/${model_label}.test.preds.txt"

#Note: 10 dimensional word vectors do very very well, severe diminishing returns even after 3-4 dimensions
#At least on dbpedia.

mkdir -p ${model_dir}

fastText="/home/jacob/Software/fastText/fasttext"

#Train model_dir
if [ ! -e "${output}.bin" ]
then
    set -x
    echo "`date`: Training fastText model"
    ${fastText} supervised -input "${train_set}" -output "${output}" -dim 100 -lr 0.1 -wordNgrams 2 -minCount 5 -bucket 1000000 -epoch 5 -thread 4 -minn 3 -maxn 6
    echo "`date`: Model training finished"
    set +x
fi

#Evaluate model_dir
echo "Evaluating ${output_bin} on ${test_set}"
${fastText} test "${output_bin}" "${test_set}"

#Predict probabilities
#This would use fastText itself, which is okay, but output format is weird
#echo "Predicting class probabilities"
#set -x
#${fastText} predict-prob ${output_bin} ${test_set_extra_classes} ${num_classes} > ${test_preds}
#set +x

#Use our custom script to format it better
if [ ! -e ${test_preds} ];
then
    set -x
    python class_probs.py ${output_bin} ${train_set} ${num_classes} > ${train_preds}
    python class_probs.py ${output_bin} ${orig_test_set} ${num_classes} > ${test_preds}
    set +x
fi

# Visualize results
set -x

echo "`date`: Visualizing results with parametric tSNE"
python text_viz_tSNE.py ${test_preds} ${orig_test_set} --classes_path "${dbpedia_data_dir}/classes.txt" --model_path "${model_dir}/dbpedia_tSNE_excl_${excl_class}.h5" --outfigurepath "${new_data_dir}/dbpedia_excl_${excl_class}.pdf" --label_prefix='__label__' --excl_class "${excl_class}"
echo "`date`: Visualizing finished"


