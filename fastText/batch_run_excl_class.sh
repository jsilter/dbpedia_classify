#!/bin/bash

for ii in {1..14}
do
    echo $ii
    ./run_fastText_training_excl_class.sh $ii
done
