import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from ori_MAE import models_mae
import models_mae_recShow
from PIL import Image
import torch.nn.functional as F
from util.datasets import read_attr

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title='',k=0):
    # image is [H, W, 3]
    if k!=0:
        title = ''
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=12)
    plt.axis('off')

    return

def show_image_list(image, title='',k=0):
    # image is [H, W, 3]
    if k!=0:
        title = ''
    assert image.shape[2] == 3
    plt.imshow()
    plt.title(title, fontsize=12)
    plt.axis('off')

    return

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):

    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model

def prepare_our_model(chkpt_dir, arch='mae_vit_base_patch16'):

    # build model
    model = getattr(models_mae_recShow, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model

def run_one_image(img,model_mae_ours):
    lab_att = read_attr(
        '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/predicate-matrix-binary.txt')  # load class attribute
    lab_att = [i[0].split(" ") for i in lab_att]
    lab_att = [list(map(int, j)) for j in lab_att]
    lab_att = torch.Tensor(lab_att)
    j = 1
    k = 0
    for x in img:
        x = torch.tensor(x)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE

        lable = [40]
        lable = torch.Tensor(lable)
        # loss, pred, mask, hash_out, cls_out
        # loss, y, mask,_,_ = model(x.float(), mask_ratio=0.5)
        y_attr,y_, mask_,_,_ ,attr= model_mae_ours(lable.long(), lab_att,x.float(), "train",mask_ratio=0.5)


        y_attr = model_mae_ours.unpatchify(y_attr)
        y_attr = torch.einsum('nchw->nhwc', y_attr).detach().cpu()

        y_ = model_mae_ours.unpatchify(y_)
        y_ = torch.einsum('nchw->nhwc', y_).detach().cpu()
        # visualize the mask
        mask = mask_.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model_mae_ours.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = model_mae_ours.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)


        im_attr_paste = x * (1 - mask) + y_attr * mask

        im_paste_ = x * (1 - mask) + y_ * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 24]

        show_image(x[0], "original",k)
        j+=1
        plt.subplot(7, 5, j)
        show_image(im_masked[0], "masked",k)
        j += 1
        plt.subplot(7, 5, j)
        show_image(im_attr_paste[0], "mix reconstruction",k)

        j += 1

        plt.subplot(7, 5, j)
        show_image(im_paste_[0], "OURS reconstruction",k)
        j += 1

        k+=1

    plt.tight_layout()
    plt.show()

# load an image
img_url = '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/JPEGImages/bobcat/bobcat_10022.jpg' # fox, from ILSVRC2012_val_00046145

img_list = [
    '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/JPEGImages/tiger/tiger_10035.jpg',
    '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/JPEGImages/horse/horse_10037.jpg',
    '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/JPEGImages/rat/rat_10012.jpg',
]
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
img_list1 = []
for i in img_list:
    img = Image.open(i)
    img = img.resize((224, 224))
    img = np.array(img) / 255
    assert img.shape == (224, 224, 3)
    img = img - imagenet_mean
    img = img / imagenet_std
    img_list1.append(img)
img = Image.open(img_url)
img = img.resize((224, 224))
img = np.array(img) / 255



# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

plt.rcParams['figure.figsize'] = [10, 8]

for i in img_list1:
    show_image(torch.tensor(i))
show_image(torch.tensor(img))
# plt.show()

# chkpt_dir = 'D:/jy/mae_jy_2/checkpoint/128/checkpoint_param_0.45720003565411277.pth'
# chkpt_dir = 'D:/jy/mae_jy_2/src/checkpoint/checkpoint_param_1.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')

chkpt_dir = '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/mae_jy_2_backup/util/vit-explain/checkpoint_param_07080.4797960003499638.pth'
model_mae_ours = prepare_our_model(chkpt_dir, 'mae_vit_base_patch16')
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
# run_one_image(img, model_mae,model_mae_ours)
run_one_image(img_list1,model_mae_ours)