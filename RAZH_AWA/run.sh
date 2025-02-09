#!/bin/bash

# 24 48 epoch 70 threthold 0.1 clu_num 9
# for i in 24 48 
# do 
#     python main_pretrain_clu.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length $i
# done

# # 64 128 epoch 50 threthold 0.2 clu_num 25
# for i in 64 128 
# do 
#     python main_pretrain_clu.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length $i
# done


# python main_pretrain_clu.py --epochs 50 --threthold 0.3 --clu_num 25 --hash_length 128

# python main_pretrain_ablation.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length 24 --mask_ratio 0.5 --gamm 0
# python main_pretrain_clu.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length 64 --gamm 0

# python main_pretrain_clu.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length 24 --gamm 0 --beta 0
# python main_pretrain_clu.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length 64 --gamm 0 --beta 0
# for i in 1 0.5 0.1 0.05
# do
#  for j in 1 0.1 0.01 0.001 0.0001
#  do
#     for k in 24 48 
#     do 
#         python main_pretrain_clu.py --epochs 70 --threthold 0.1 --clu_num 9 --hash_length $k --alpha $i --gamm $j
#     done

#     for k in 64 128 
#     do 
#         python main_pretrain_clu.py --epochs 50 --threthold 0.2 --clu_num 25 --hash_length $k --alpha $i --gamm $j
#     done
#  done
# done

for i in $(seq 0 49); do

python vit-explain-main/vit_explain.py --category_index $i
done