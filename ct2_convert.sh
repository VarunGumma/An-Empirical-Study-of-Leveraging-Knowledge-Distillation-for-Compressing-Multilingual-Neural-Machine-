#/bin/bash
echo `date`

data_dir=$1
model_name=$2

ct2-fairseq-converter \
--data_dir ${data_dir}/final_bin \
--model_path ${data_dir}/${model_name}/checkpoint_best.pt \
--output_dir ${data_dir}/${model_name}/ct2-converted \
--source_lang SRC \
--target_lang TGT \
--quantization float16 \
--user_dir model_configs \
--force
