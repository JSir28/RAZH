import torch
import torch.nn as nn
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn.functional as F

class My_Loss(nn.Module):
    def __init__(self, num_classes, hash_code_length, mixup_fn, smoothing, alph, beta, gamm,unseen_class):
        super().__init__()
        self.num_classes = num_classes
        self.hash_code_length = hash_code_length
        self.alph = alph
        self.beta = beta
        self.gamm = gamm

        self.unseenclass = unseen_class
        self.attr = self.get_attr()
        if mixup_fn is not None:
            self.classify_loss_fun = SoftTargetCrossEntropy()
        elif smoothing > 0.:
            self.classify_loss_fun = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self.classify_loss_fun = torch.nn.CrossEntropyLoss()
    # def get_unseenclass(self):
    #     unseen_classes = open(
    #         'D:/jy/datasets/AWA2/Animals_with_Attributes2/testclasses.txt').readlines()
    #     all_classes = open('D:/jy/datasets/AWA2/Animals_with_Attributes2/classes.txt').readlines()
    #     all_classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in all_classes]
    #     unseen_classes = [i.replace('\n', '') for i in unseen_classes]
    #
    #     target = [all_classes.index(i) for i in unseen_classes]
    #     target = torch.tensor(target)
    #     return target

    def get_attr(self):
        attr = open(
            'D:/jy/datasets/AWA2/Animals_with_Attributes2/predicate-matrix-binary.txt').readlines()
        return attr

    def dcl(self, hash_out, target, tep=0.5):
        # target = F.one_hot(target, 50).float()
        target = self.attr[target]
        sim = (torch.einsum('ij,jk->ik', target, target.t()) > 0).float()
        bate = sim / torch.sum(sim)
        inner_product = torch.einsum('ij,jk->ik', hash_out, hash_out.t())
        dis = ((inner_product + 32.0) / 2)
        dis_exp = torch.exp(dis) / tep

        loss = -bate * torch.log(dis_exp / torch.sum(dis_exp))

        return loss.sum()

    def hash_loss(self, hash_out, target):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        one_hot = torch.nn.functional.one_hot(target.to(torch.int64), self.num_classes)
        one_hot = one_hot.float()
        Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()


        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)

        mask_positive = Sim > 0
        mask_negative = Sim <= 0
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1
        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix
        count = hash_out.shape[0]

        return pair_loss.sum() / 2 / count

    # def quanti_loss(self, hash_out):
    #     regular_term = (hash_out - hash_out.sign()).pow(2).mean()
    #     return regular_term

    def quanti_loss(self, out):
        b_matrix = torch.sign(out)
        temp = torch.einsum('ij,jk->ik', out, out.t())
        temp1 = torch.einsum('ij,jk->ik', b_matrix, b_matrix.t())
        q_loss = temp - temp1
        q_loss = torch.abs(q_loss)
        loss = torch.exp(q_loss / out.shape[1])

        # return loss.sum()
        return loss.sum() / out.shape[0] / out.shape[0]

    def compute_loss_Self_Calibrate(self, cls_out):
        S_pp = cls_out
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def forward(self, hash_out, cls_out, cls_out_attr, hash_out_attr,target):
        cls_loss = self.classify_loss_fun(cls_out, target)+self.classify_loss_fun(cls_out_attr, target)
        hash_loss = self.hash_loss(hash_out, target)
        quanti_loss = self.quanti_loss(hash_out)
        cla_loss = self.compute_loss_Self_Calibrate(cls_out=cls_out)
        loss = self.gamm * (cls_loss+cla_loss) + self.alph * (hash_loss)
        return hash_loss, quanti_loss, cls_loss, loss


class My_Loss_eval(nn.Module):
    def __init__(self, num_classes, hash_code_length, alph, beta, gamm):
        super().__init__()
        self.num_classes = num_classes
        self.hash_code_length = hash_code_length
        self.alph = alph
        self.beta = beta
        self.gamm = gamm
        self.classify_loss_fun = torch.nn.CrossEntropyLoss()

    def hash_loss(self, hash_out, target):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        one_hot = one_hot.float()
        Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()


        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)

        mask_positive = Sim > 0
        mask_negative = Sim <= 0
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1
        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix
        count = (hash_out.shape[0] * (hash_out.shape[0] - 1) / 2)

        return pair_loss.sum() / 2 / count

    def quanti_loss(self, hash_out):
        regular_term = (hash_out - hash_out.sign()).pow(2).mean()
        return regular_term

    def forward(self, hash_out, cls_out, target):
        cls_loss = self.classify_loss_fun(cls_out, target)
        hash_loss = self.hash_loss(hash_out, target)
        quanti_loss = self.quanti_loss(hash_out)
        loss = self.gamm * cls_loss + self.alph * hash_loss
        return hash_loss, quanti_loss, cls_loss, loss
