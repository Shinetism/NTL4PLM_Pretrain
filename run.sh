TOTAL_NUM_UPDATES=12500  # 12500/17 steps for pretraining
WARMUP_UPDATES=750      # 6 percent of the number of updates
PEAK_LR=5e-04                # Peak LR for polynomial LR scheduler.
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITION=512    # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=128          # Increase the batch size 16x
ROBERTA_PATH=model-bin/roberta-cs-base
SAVE_DIR=model-bin/roberta-cs-bio-17G

SOURCE_DOMAIN=data-bin/corpus/cs_17G
AUXILIARY_DOMAIN=data-bin/corpus/bio_17G

if [ ! -d "$SAVE_DIR" ]; then
    mkdir $SAVE_DIR
fi

if [ -f "$SAVE_DIR/checkpoint_last.pt" ]; then
    RESTORE_FILE=$SAVE_DIR/checkpoint_last.pt
else
    RESTORE_FILE=$ROBERTA_PATH/model.pt
    EXTRA="--reset-optimizer --reset-dataloader --reset-meters"
fi

CUDA_VISIBLE_DEVICES=1,2,3,6 python3 train.py $SOURCE_DOMAIN $AUXILIARY_DOMAIN \
    --task ntl_pretrain \
    --criterion ntl \
    --arch roberta_base \
    --restore-file $RESTORE_FILE \
    --seed 314 \
    --save-dir $SAVE_DIR \
    --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_NUM_UPDATES --log-format tqdm --log-interval 1 --tensorboard-logdir $SAVE_DIR/tensorboard \
    --skip-invalid-size-inputs-valid-test \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --fixed-validation-seed 0 --ddp-backend no_c10d --save-interval-updates 100 \
    --find-unused-parameters $EXTRA | tee -a $SAVE_DIR/log.txt;

