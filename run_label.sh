#dataset=$1
#device=$2
#
#[ -z "${dataset}" ] && dataset="cora"
#[ -z "${device}" ] && device=-1

#for k in 24 64
#    do
#      python ./main_pretrain_Lr.py \
#      --hash_length $k
#    done



for k in 24 48
do
  python ./main_pretrain_label.py  --hash_length $k
done


#for i in 24 64
#  do
#    python ./main_pretrain_mae_cifar10.py \
#    --hash_length $i
#  done
#
#for i in 24 64
#  do
#    python ./main_pretrain_mae_cub.py \
#    --hash_length $i --dataset_type cub
#  done
#
#for i in 48 128
#  do
#    python ./main_pretrain_mae_cifar10.py \
#    --hash_length $i
#  done
#
#for i in 48 128
#  do
#    python ./main_pretrain_mae_cub.py \
#    --hash_length $i --dataset_type cub
#  done



#for j in 1 2 3
#do
#  for i in 24 48 64 128
#  do
#    python ./main_pretrain_cub.py \
#    --model mae_vit_base_patch16 \
#    --retrain E:/jy/backup/mae_jy_2/mae_pretrain_vit_base.pth \
#    --hash_length $i --lr 1e-5
#  done
#done
#
#for j in 1 2 3
#do
#  for i in 24 48 64 128
#  do
#    python E:/jy/backup/mae_jy_1/main_pretrain_cub.py \
#    --model mae_vit_base_patch16 \
#    --retrain E:/jy/backup/mae_jy_2/mae_pretrain_vit_base.pth \
#    --hash_length $i
#  done
#done