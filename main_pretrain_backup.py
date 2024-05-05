# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import pickle
from timm.utils import accuracy, AverageMeter
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import os
import time
from pathlib import Path

from util.datasets import read_attr
from util.evaluation import presicion_and_recall
from util.tools import *
from util.datasets_jy import *
from util.pos_embed import interpolate_pos_embed
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_cls
from engine_pretrain import train_one_epoch

device = torch.device('cuda')


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=105, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--hash_dir', default='./hash_dir', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--gamm', default=1e-1, type=float)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of mod el to train')
    parser.add_argument('--retrain', default='./mae_pretrain_vit_base.pth', type=str,
                        help='Name of model to train')
    parser.add_argument('--hash_length', default=64, type=int,
                        help='Name of model to train')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=6e-6, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='awa2', type=str,
                        help='dataset')
    parser.add_argument('--dataset_type', default='awa2', type=str,
                        help='dataset')
    parser.add_argument('--crop_size', default='224', type=str,
                        help='dataset')
    parser.add_argument('--classnum', default=50, type=int,
                        help='classnum')
    parser.add_argument('--TOP_K', default=[5000], type=int,
                        help='TOP_K')
    parser.add_argument('--w2vpath', default='./word2vec/AWA2_attribute.pkl',
                        type=str,
                        help='dataset')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_unseenclass():
    unseen_classes = open(
        '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/testclasses.txt').readlines()
    all_classes = open(
        '/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/datasets/AWA2/Animals_with_Attributes2/classes.txt').readlines()
    all_classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in all_classes]
    unseen_classes = [i.replace('\n', '') for i in unseen_classes]

    target = [all_classes.index(i) for i in unseen_classes]
    target = torch.tensor(target)
    return target


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_torch(0)

    cudnn.benchmark = True

    args.log_dir = args.log_dir + "/" + str(args.mask_ratio) + "/" + str(time.time()) + "_" + str(
        args.hash_length) + "_" + str(args.dataset_type) + "_" + str(args.alpha) + "_" + str(args.gamm)
    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # loading w2v
    # src_att = torch.load(args.w2vpath)
    # src_att = torch.from_numpy(np.load(args.w2vpath)) #loading attribute w2v
    with open(args.w2vpath, 'rb') as f:
        src_att = torch.tensor(pickle.load(f))
    src_att.to(device)
    lab_att = read_attr(
        os.path.join(args.data_path, 'Animals_with_Attributes2/predicate-matrix-binary.txt'))  # load class attribute
    lab_att = [i[0].split(" ") for i in lab_att]
    lab_att = [list(map(int, j)) for j in lab_att]
    lab_att = torch.Tensor(lab_att)

    # data_loader_train, test_loader,database_loader, num_train, num_test, num_database = get_data(args)
    # database_loader = get_database_data(test_loader, args)
    data_loader_train, num_train = get_train_data(args)
    unseen = get_unseenclass()
    print(num_train)

    # define the model
    model = models_mae_cls.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, hash_length=args.hash_length,
                                                unseen_classes=unseen, alpha=args.alpha, gamm=args.gamm)

    if args.retrain:
        checkpoint = torch.load(args.retrain, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.retrain)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    # criterion = torch.nn.CrossEntropyLoss()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    max_map = 0.0
    max_precison = 0.0
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        print("new attr")
        model.attr = src_att
        train_stats = train_one_epoch(
            model, data_loader_train, src_att, lab_att,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        Map = 0

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if epoch > 50 or epoch + 1 == args.epochs or epoch > 90:
            acc1, attr_acc1, acc5, attr_acc5, loss, Map, Precision = validata(args, data_loader_train, src_att, lab_att,
                                                                              model, epoch,
                                                                              max_map, log_writer=log_writer)
            max_map = max(max_map, Map)
            max_precison = max(max_precison, Precision)
            print(f'Max map:{max_map:.6f} Max Precision:{max_precison[0]:.6f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def validata(args, data_loader_train, src_att, lab_att, model, epoch, max_map, log_writer):
    test_loader, database_loader, num_test, num_database = get_query_data(args, data_loader_train)
    print("test:", num_test)
    print("database:", num_database)
    model.eval()
    if (epoch > 50 and epoch < 65) or epoch + 1 == args.epochs or epoch > 90:
        query_label_matrix = np.empty(shape=(0,))
        query_label_matrix1 = np.empty(shape=(0, 50,))
        query_hash_matrix = np.empty(shape=(0, args.hash_length))
        database_label_matrix = np.empty(shape=(0,))
        database_label_matrix1 = np.empty(shape=(0, 50,))
        database_hash_matrix = np.empty(shape=(0, args.hash_length))

    loss_meter = AverageMeter()
    map_list = []
    map_list1 = []
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    attr_acc1_meter = AverageMeter()
    attr_acc5_meter = AverageMeter()
    for sample in enumerate(test_loader):
        idx = sample[0]
        images = sample[1][0]
        target = sample[1][1]
        images = images.to(device)
        target = target.to(device)
        src_att = src_att.to(device, non_blocking=True)
        lab_att = lab_att.to(device, non_blocking=True)
        with torch.no_grad():
            loss, _, _, hash_out, cls_out = model(target, lab_att, images, "test", mask_ratio=args.mask_ratio)

        if (epoch > 50 and epoch < 65) or epoch + 1 == args.epochs or epoch > 90:
            hash_code = torch.sign(hash_out)
            hash_code = hash_code.cpu().numpy()
            one_hot_label = F.one_hot(target, 50)
            query_label_matrix = np.concatenate((query_label_matrix, target.cpu().numpy()), axis=0)
            query_label_matrix1 = np.concatenate((query_label_matrix1, one_hot_label.cpu().numpy()), axis=0)
            query_hash_matrix = np.concatenate((query_hash_matrix, hash_code), axis=0)
        acc1, acc5 = accuracy(cls_out, target, topk=(1, 5))

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

    if (epoch > 50 and epoch < 65) or epoch + 1 == args.epochs or epoch > 90:
        database_acc1_meter = AverageMeter()
        database_acc5_meter = AverageMeter()
        loop = tqdm(enumerate(database_loader), total=len(database_loader))
        for step, sample in loop:
            images = sample[0]
            target = sample[1]
            images = images.to(device)
            target = target.to(device)
            with torch.no_grad():
                loss, _, _, hash_out, cls_out = model(target, lab_att, images, "test", mask_ratio=args.mask_ratio)

            if (epoch > 50 and epoch < 65) or epoch + 1 == args.epochs or epoch > 90:
                hash_code = torch.sign(hash_out)
                hash_code = hash_code.cpu().numpy()

                one_hot_label = F.one_hot(target, 50)
                database_label_matrix = np.concatenate((database_label_matrix, target.cpu().numpy()), axis=0)
                database_label_matrix1 = np.concatenate((database_label_matrix1, one_hot_label.cpu().numpy()), axis=0)
                database_hash_matrix = np.concatenate((database_hash_matrix, hash_code), axis=0)
                acc1, acc5 = accuracy(cls_out, target, topk=(1, 5))

                database_acc1_meter.update(acc1.item(), target.size(0))
                database_acc5_meter.update(acc5.item(), target.size(0))

        Map, Recall = mean_average_precision_R(database_hash_matrix, query_hash_matrix, database_label_matrix,
                                               query_label_matrix, args.TOP_K[0], args.classnum)

        presicion1, recall1, map1, wap1, acg1, = presicion_and_recall(query_hash_matrix, database_hash_matrix,
                                                                      args.TOP_K, query_label_matrix1,
                                                                      database_label_matrix1)
        map_list.append({"map": Map, "Recall": Recall})

        if epoch + 1 == args.epochs:
            print(map_list)
            print(map_list1)
        os.makedirs(args.hash_dir + '/hash_0708/', exist_ok=True)
        np.savez(
            args.hash_dir + '/hash_0708/map_' + str(Map) + '_' + str(epoch) + '_query_hash_label',
            label=query_label_matrix, hash_code=query_hash_matrix)
        np.savez(
            args.hash_dir + '/hash_0708/map_' + str(Map) + '_' + str(epoch) + '_database_hash_label',
            label=database_label_matrix, hash_code=database_hash_matrix)
        log_writer.add_scalar('Map', Map, epoch)
        log_writer.add_scalar('Recall', Recall, epoch)
        log_writer.add_scalar('Presicion', presicion1[0], epoch)
        print("Map:", Map, "Recall:", Recall)
        print("Map1:", map1, "Recall1:", recall1, "Precision:", presicion1)

        map_stats = {'map': str(Map),
                     'epoch': str(epoch), 'hash_lenght': str(args.hash_length), }
        with open(os.path.join(args.log_dir, "map.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(map_stats) + "\n")

        # torch.save(model, "D:/jy/mae_jy_2/checkpoint/" + str(args.hash_length) + "/checkpoint_0708" + str(Map) + ".pth")
        # torch.save(model.state_dict(), "D:/jy/mae_jy_2/checkpoint/" + str(args.hash_length) + "/checkpoint_param_0708" + str(Map) + ".pth")
        return acc1_meter.avg, attr_acc1_meter.avg, acc5_meter.avg, attr_acc5_meter.avg, loss_meter.avg, Map, presicion1
    return acc1_meter.avg, attr_acc1_meter.avg, acc5_meter.avg, attr_acc5_meter.avg, loss_meter.avg, 0.0, 0.0


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
