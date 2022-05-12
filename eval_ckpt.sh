CUDA_VISIBLE_DEVICES=0,1 python eval_ckpt.py data-bin/corpus/cs_17G --path model-bin/roberta-cs-bio-17G/checkpoint_best.pt --task masked_lm --batch-size 64
