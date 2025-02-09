#!/bin/bash

#python main_pretrain.py --hash_length 24 --alpha 0.05 --gamm 1.0
# python main_pretrain.py --hash_length 24
# python main_pretrain.py --hash_length 48
# python main_pretrain.py --hash_length 64 
# python main_pretrain.py --hash_length 128
#python main_pretrain.py --hash_length 64 --alpha 0.05 --gamm 1.0
#python main_pretrain.py --hash_length 128 --alpha 0.01 --gamm 0.01

# for i in 1 5e-1 1e-1 5e-2 1e-2
# do
#     for j in 1 1e-1 1e-2 1e-3 1e-4
#     do 
#         python main_pretrain.py --hash_length 48 --alpha $i --gamm $j
#     done
# done    

# for m in 0.25 0.75
# do
#     for h in 24 64 
#     do
#         python main_pretrain.py --hash_length $h --mask_ratio $m
#     done
# done

for h in 24 64 
    do
        python main_pretrain.py --hash_length $h --gamm 0
    done