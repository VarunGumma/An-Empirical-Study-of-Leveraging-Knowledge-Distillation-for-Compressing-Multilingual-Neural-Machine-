#!/bin/bash
echo `date`

echo -e "[INFO]\tremoving old data"
rm -rf ../data/
echo -e "[INFO]\tunzipping compressed data"
unzip -q ../data.zip -d ..
echo -e "[INFO]\tcreating randomized variations"
python3 randomized_sampling.py ../data/v2

echo `date`

for idx in 0 10 20 30 40 50 60 70 80 90 100; do 
    bash prepare_distilled_data.sh ../data/v2_$idx ../data/benchmarks ../data/v2_${idx}_binarized
    echo -e "********************************************************************\n"
done 