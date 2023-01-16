#!/bin/bash
echo `date`

bash joint_translate.sh ../data_raw/v2/en-as/sample.as ../data_raw/v2/en-as/sample_trans.en as en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-bn/sample.bn ../data_raw/v2/en-bn/sample_trans.en bn en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-gu/sample.gu ../data_raw/v2/en-gu/sample_trans.en gu en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-hi/sample.hi ../data_raw/v2/en-hi/sample_trans.en hi en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-kn/sample.kn ../data_raw/v2/en-kn/sample_trans.en kn en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-ml/sample.ml ../data_raw/v2/en-ml/sample_trans.en ml en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-mr/sample.mr ../data_raw/v2/en-mr/sample_trans.en mr en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-or/sample.or ../data_raw/v2/en-or/sample_trans.en or en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-pa/sample.pa ../data_raw/v2/en-pa/sample_trans.en pa en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-ta/sample.ta ../data_raw/v2/en-ta/sample_trans.en ta en checkpoints/it ../data_dir/v2_indic_en_bin
bash joint_translate.sh ../data_raw/v2/en-te/sample.te ../data_raw/v2/en-te/sample_trans.en te en checkpoints/it ../data_dir/v2_indic_en_bin