#!/bin/bash

# 24 48 epoch 70 threthold 0.1 clu_num 9
# for i in 24
# do 
# for m in 0.75
#     do
#         python main_pretrain_clu.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length $i --mask_ratio $m
#     done
# done

# 64 128 epoch 50 threthold 0.2 clu_num 25
# for i in 64
# do 
# for m in 0.25 0.75
#     do
#         python main_pretrain_clu.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length $i --mask_ratio $m
#     done
# done
# python main_pretrain_ablation.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length 24 --mask_ratio 0.5 --alpha 0 --gamm 0
# python main_pretrain_ablation.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length 64 --mask_ratio 0.5 --alpha 0 --gamm 0
# python main_pretrain_ablation.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length 24 --mask_ratio 0.5 --gamm 0
# python main_pretrain_ablation.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length 64 --mask_ratio 0.5 --gamm 0

python main_pretrain_ablation.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length 24
python main_pretrain_ablation.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length 64