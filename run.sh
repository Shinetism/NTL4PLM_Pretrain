TOTAL_NUM_UPDATES=12500  # 12500 steps for pretraining
WARMUP_UPDATES=750      # 6 percent of the number of updates
PEAK_LR=5e-04                # Peak LR for polynomial LR scheduler.
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITION=512    # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=128          # Increase the batch size 16x
ROBERTA_PATH=model-bin/roberta-wb-base

SOURCE_DOMAIN=data-bin/corpus/cs_1G
AUXILIARY_DOMAIN=data-bin/corpus/bio_1G

CUDA_VISIBLE_DEVICES=5 python3 train.py $SOURCE_DOMAIN $AUXILIARY_DOMAIN \
    --task ntl_pretrain \
    --criterion ntl \
    --arch roberta_base \
    --pretrained-model-name-or-path model-bin/roberta-wb-base \
    --seed 314 \
    --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES --log-format simple --log-interval 1\
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --find-unused-parameters;

## TODO: save best model with minimal loss