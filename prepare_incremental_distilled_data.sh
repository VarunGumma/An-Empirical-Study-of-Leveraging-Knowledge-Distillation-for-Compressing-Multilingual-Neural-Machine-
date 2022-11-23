#!/bin/bash
echo `date`
data_dir=$1

echo -e "[INFO]\tcreating randomized variations"
python3 randomized_sampling.py data_dir/v2

echo `date`

for idx in 0 10 20 30 40 50 60 70 80 90 100; do 
    # idx denotes percentage of original targets present in the mixed sample
    bash prepare_distilled_data.sh $data_dir/v2_$idx $data_dir/benchmarks $data_dir/v2_${idx}_binarized
    # remove unnecessary files from the binarized directories to save space
    if [ $idx -gt 0 ]; then
        rm $data_dir/v2_${idx}_binarized/final_bin/valid.* $data_dir/v2_${idx}_binarized/final_bin/test.* 
    fi
    mv $data_dir/v2_${idx}_binarized/final_bin/* $data_dir/v2_${idx}_binarized/
    rm -rf $data_dir/v2_${idx}_binarized/vocab $data_dir/v2_${idx}_binarized/final_bin
    rm $data_dir/v2_${idx}_binarized/preprocess.log
done 

tar -czvf $data_bin_dir/incrementally_distilled_samanantar_binarzed_data.tar.gz $data_bin_dir/*