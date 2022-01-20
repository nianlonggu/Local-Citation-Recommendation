#!/bin/bash

rm $4
count=0
seed=$RANDOM
for (( start=0; start<$1; start += $2))
    do
        python compute_papers_embedding.py -shuffle_seed $seed -encoder_gpu_list $count -config_file_path $3 -start $start -size $2 -signal_file $4  &
        ((count=count+1))
    done
python merge_papers_embedding.py -num_process $count -signal_file $4 -slice_name_prefix $5 -save_name $6
rm $4
