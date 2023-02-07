for lang in as bn gu hi kn ml mr or pa ta te; do
    if [[ "$lang" == as ]]; then
        warmup=1000
        update_freq=1
        distributed_world_size=1
    elif [[ "$lang" == gu ]]; then
        warmup=2000
        update_freq=1
        distributed_world_size=4
    elif [[ "$lang" == or ]]; then
        warmup=1600
        update_freq=1
        distributed_world_size=4
    else
        warmup=4000
        update_freq=1
        distributed_world_size=4
    fi

    fairseq-train ../../../data_bin/bilingual_wo_transliteration/distilled/$lang/final_bin \
    --max-source-positions 210 \
    --max-target-positions 210 \
    --max-update 1000000 \
    --max-tokens 2048 \
    --arch transformer \
    --activation-fn gelu \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --layernorm-embedding \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --source-lang SRC \
    --target-lang TGT \
    --lr-scheduler inverse_sqrt \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-09 \
    --lr 7e-4 \
    --warmup-updates $warmup \
    --save-dir ../../checkpoints/distilled_bilingual/${lang}_wo_transliteration \
    --save-interval 1 \
    --save-interval-updates $warmup \
    --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --patience 5 \
    --skip-invalid-size-inputs-valid-test \
    --update-freq $update_freq \
    --distributed-world-size $distributed_world_size \
    --num-workers 8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --memory-efficient-fp16 \
    --wandb-project Indic-En-Distillation

    echo "------------------------------------------------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------------------------------------------------"
done
