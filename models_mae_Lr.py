# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import pickle
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from my_loss_abl import My_Loss
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=24, num_heads=16, hash_length=128,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, w2v_dim=300,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_classes=50, attr=None,alpha=1e-1,gamm=1e-1,
                 unseen_classes=None):
        super().__init__()

        self.alpha = alpha
        self.gamm = gamm
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.hash_length = hash_length
        self.embed_dim = embed_dim
        self.hash_layer = nn.Linear(embed_dim, self.hash_length)
        self.head = nn.Linear(hash_length, num_classes) if num_classes > 0 else nn.Identity()
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # latent align specifics
        self.align_layer = nn.Linear(embed_dim, w2v_dim, bias=True)
        self.unalign_layer = nn.Linear(w2v_dim, embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.V = nn.Parameter(attr)
        self.embed_attr = nn.Sequential(nn.Linear(w2v_dim, w2v_dim), nn.Linear(w2v_dim, w2v_dim))
        self.initialize_weights()
        self.unseen_classes = unseen_classes
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, src_att, lab_att, labels, mode, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        ids_replace = ids_shuffle[:, len_keep:]
        x_placed = torch.gather(x, dim=1, index=ids_replace.unsqueeze(-1).repeat(1, 1, D))

        x_ = self.forward_attribute(x_placed, attr=src_att, lab_att=lab_att, labels=labels, mode=mode)

        x_attr = torch.cat((x_masked, x_), dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_attr, mask, ids_restore

    # class AttentionPooling(nn.Module):
    #     def __init__(self, in_dim):
    #         super().__init__()
    #         self.cls_vec = nn.Parameter(torch.randn(in_dim))
    #         self.fc = nn.Linear(in_dim, in_dim)
    #         self.softmax = nn.Softmax(-1)

    # def AttentionPooling(self, x,cls):
    #     cls_ = cls.permute(0,2,1)
    #     # weights = x@cls_.T
    #     weights = torch.matmul(x, cls_)
    #     weights = self.softmax(weights.view(x.shape[0], -1))
    #     # if self.training==False:
    #     #     top = torch.topk(weights, int(weights.shape[1]/2), dim=-1)
    #     #     idx = top[1]
    #     #     weights = top[0]
    #     #     x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, x.shape[2]))
    #
    #     x = torch.bmm(weights.unsqueeze(-2),x).squeeze()
    #     x = x + cls.squeeze()
    #     x = self.fc(x)
    #     x = x + cls.squeeze()
    #     return x

    def forward_encoder(self, x, src_att, lab_att, labels, mode, mask_ratio):  # 图片 掩码率
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        mask = None
        ids_restore = None
        x = x + self.pos_embed[:, 1:, :]  # 加上全零的位置embedding,x直接加上0

        # masking: length -> length * mask_ratio
        if mode == "train":
            x_masked, x_attr, mask, ids_restore = self.random_masking(x, src_att, lab_att, labels, mode, mask_ratio)
        else:
            x_attr = None
            x_masked = x

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)
        x_out = self.avgpool(x_masked.transpose(1, 2))  # B C 1
        x_out = torch.flatten(x_out, 1)
        hash_out = self.hash_layer(x_out)
        cls_out = self.head(hash_out)
        hash_out = torch.tanh(hash_out)


        # hash_out, cls_out = self.get_out(cls_token, x_masked)
        if self.training == True:
            cls_tokens = cls_token.expand(x_attr.shape[0], -1, -1)
            x_attr = torch.cat((cls_tokens, x_attr), dim=1)
            for blk in self.blocks:
                x_attr = blk(x_attr)
            x_attr = self.norm(x_attr)
            x_out = self.avgpool(x_attr.transpose(1, 2))  # B C 1
            x_out = torch.flatten(x_out, 1)
            hash_out_attr = self.hash_layer(x_out)
            cls_out_attr = self.head(hash_out_attr)
            hash_out_attr = torch.tanh(hash_out_attr)
        else:
            hash_out_attr = None
            cls_out_attr = None
        return x_masked, mask, ids_restore, cls_out, hash_out, x_attr, cls_out_attr, hash_out_attr


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_attribute(self, x, attr, lab_att, labels, mode):
        # cls_token = x[:, 0, :]
        # cls_token = cls_token.unsqueeze(dim=1)
        x_ori = x
        attr_nor = 1. * attr / (torch.norm(attr, 2, -1, keepdim=True).expand_as(attr) + 1e-12)
        x = self.align_layer(x)  # align to w2v
        # attr = attr.T.float()
        x_nor = 1. * x / (torch.norm(x, 2, -1, keepdim=True).expand_as(x) + 1e-12)
        score = torch.einsum("abc, cd->abd", (x_nor, attr_nor.T.float()))

        # 归一化
        # a = torch.norm(attr,2,-2,keepdim=True).expand_as(attr)
        # attr_nor = 1.*attr/(torch.norm(attr,2,-2,keepdim=True).expand_as(attr)+1e-12)
        # 余弦相似度
        cos_attr = torch.mm(attr_nor, attr_nor.T)
        sort_score = torch.topk(score, 1, dim=2)[1]
        # print(sort_score)
        # la = torch.unique(sort_score)
        # print(la)
        # sort_score_content = torch.topk(score, 1, dim=2)[0]

        attr_label = lab_att[labels]
        # index = torch.nonzero(attr_label==1)
        # print(index)
        attr_label_max = attr_label.detach().unsqueeze(dim=1).repeat(1, score.shape[1], 1)

        attr_label_score = torch.mul(attr_label_max.cuda(), score.cuda())
        label_attr_sort_score = torch.topk(attr_label_score, 1, dim=2)

        # relation_score = cos_attr[label_attr_sort_score[1], sort_score]
        #
        # relation_score_top = torch.topk(relation_score, 6, dim=2)
        # attr_tmp_index = relation_score_top[1]
        # attr_index = torch.gather(sort_score, dim=2, index=attr_tmp_index)
        # attr_index = attr_index[:,:,1:]
        # rela = relation_score_top[0][:,:,1:]
        # # attr_index = sort_score[attr_tmp_index]
        #
        # a = attr[attr_index]
        # # attr_index = sort_score[attr_tmp_index]
        #
        # #        a = attr[attr_index]
        # attr_sim = torch.sum(torch.mul(a, rela.unsqueeze(dim=3).expand_as(a)), dim=2)

        attr_decoder = attr[label_attr_sort_score[1].squeeze()]
        # attr_decoder = attr[label_attr_sort_score[1].squeeze()]
        attr_decoder = attr_decoder.to(torch.float32)
        attr_decoder = self.unalign_layer(attr_decoder)

        # mask = sort_score_content<=0.01
        # attr_decoder = torch.where(mask,x_ori.float(),attr_decoder.float())
        return attr_decoder

    def forward_loss(self, imgs, pred,att_pred,mask, labels, cls_out, hash_out, cls_out_attr, hash_out_attr):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        criterion = My_Loss(self.num_classes, self.hash_length, mixup_fn=None, smoothing=0,
                            alph=self.alpha, beta=1, gamm=1.0,unseen_class=self.unseen_classes)
        hash_loss, quanti_loss, cls_loss, loss1 = criterion(hash_out, cls_out,labels)
        _, _, _, loss1_attr = criterion(hash_out_attr,cls_out_attr,labels)
        # hash1 = torch.sign(hash_out_attr)
        # hash2 = torch.sign(hash_out)
        # loss3 =(hash_out - hash_out_attr).pow(2).mean()
        # loss3 = torch.abs(torch.sum(hash_out_attr - hash_out)/ hash_out.shape[0])
        # loss3_ = torch.abs(torch.sum(hash_out_attr - hash_out)).mean()
        # loss3 = torch.abs(0.5*torch.sum(hash_out_attr - hash_out) / hash_out.shape[0])
        # loss3=0.0
        # loss3 = 0.1*self.dcl(hash_out,hash_out_attr,labels)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        loss2 = (target-att_pred) **2
        loss2 = loss2.mean(dim=-1)  # [N, L], mean loss per patch

        loss2 = (loss2 * mask).sum() / mask.sum()  # mean loss on removed patches


        rec_loss = self.gamm*(loss+loss2)
        # loss3 = loss =loss2 =0.0
        return rec_loss+loss1+loss1_attr

    def forward(self,labels, lab_att,imgs, mode,mask_ratio=0.25):
        attr = self.embed_attr(self.V.float())
        latent, mask, ids_restore, cls_out, hash_out, latent_attr, cls_out_attr, hash_out_attr = self.forward_encoder(imgs, attr, lab_att, labels, mode,mask_ratio)
        # latent, mask, ids_restore,cls_out, hash_out = self.forward_encoder(imgs, mask_ratio)
        if self.training == False:
            return 0.0, None, mask, hash_out, cls_out
        # att_latent = self.forward_attribute(latent,lab_att,labels)
        att_pred = self.forward_decoder(latent_attr, ids_restore)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
   
        loss = self.forward_loss(imgs, pred,att_pred, mask, labels,cls_out, hash_out, cls_out_attr, hash_out_attr)

        return loss, pred, mask, hash_out, cls_out


def mae_vit_base_patch16_dec512d8b(**kwargs):
    with open('/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/mae_jy_2_backup/word2vec/AWA2_attribute.pkl', 'rb') as f:
        w2v_att = torch.tensor(pickle.load(f))
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), attr=w2v_att, **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
