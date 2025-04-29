#!/bin/bash

# for i in 1 0.5 0.1 0.05
# do
#   for j in 1 0.1 0.01 0.001 0.0001
#   do
#       python main_pretrain.py --hash_length 128 --alpha $i --gamm $j
#   done
# done

# python main_pretrain.py --hash_length 2 --mask_ratio $i


for h in 24 64
do
  python main_pretrain_abalation.py --hash_length $h
done 



#python main_pretrain.py --hash_length 24 --alpha 0.05 --gamm 1
#python main_pretrain.py --hash_length 48 --alpha 1 --gamm 0.001
#python main_pretrain.py --hash_length 64 --alpha 0.05 --gamm 1
#python main_pretrain.py --hash_length 128 --alpha 1 --gamm 0.0001