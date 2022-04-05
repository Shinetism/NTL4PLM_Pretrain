# NTL4PLM_Pretrain

## Install Environment

```
conda create -n NTL4PLM_Pretrain python=3.7
conda activate NTL4PLM_Pretrain
pip install fairseq==0.9.0 requests==2.27.1 tensorboardX==2.5
```

## Download Pretrained Model

Copy the roberta-wb-base model to `model-bin/roberta-wb-base/model.pt`
and the dictionary to `model-bin/roberta-wb-base/dict.txt`.

Copy the roberta-cs-base model to `model-bin/roberta-cs-base/model.pt`
and copy the dictionary from roberta-wb-base to `model-bin/roberta-cs-base/dict.txt`.


## Download Data

Copy the 17G cs & bio data to `data-bin/corpus/cs_17G` and
`data-bin/corpus/bio_17G`.

## Training

```
ulimit -n 2048
bash run.sh
```