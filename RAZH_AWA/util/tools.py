import os

import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import random

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "./dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "./dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/COCO/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "./dataset/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R




class ImageList(object):

    def __init__(self, data_path, dataset,classes,image_list, transform):
        self.dataset = dataset
        self.data_path = data_path+'Animals_with_Attributes2/JPEGImages/'
        self.classes = classes
        self.imgs = self.load_image(image_list, dataset)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        target = [img.split("/")[-1].split("_")[0] for img in self.imgs]
        self.target = [self.classes.index(i) for i in target]


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-1].split("_")[0]
        target = self.classes.index(target)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def load_image(self,class_name, dataset):
        list = []
        select_size = 100
        for i in class_name:
            i = i.replace('\n', '')
            data_path_item = self.data_path+i
            image_name = os.listdir(data_path_item)
            if dataset=="train_set":
                if len(image_name)<=select_size:
                    image_select = random.sample(image_name, len(image_name)-1)
                    select_size = select_size+(100-len(image_name)+1)
                else:
                    image_select = random.sample(image_name, select_size)
                    select_size = 100

            else:
                image_select = random.sample(image_name, 100)
            for j in image_select:
                list.append(os.path.join(data_path_item+"/",j))
        return list


class ImageList_database(object):

    def __init__(self, data_path, test_loader,train_loader,classes,image_list, transform):
        self.loader = test_loader
        self.data_path = data_path+'Animals_with_Attributes2/JPEGImages/'
        self.classes = classes
        self.imgs = self.load_image(image_list, test_loader,train_loader)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-1].split("_")[0]
        target = self.classes.index(target)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def load_image(self,class_name, loader, train_loader):
        list = []
        imgs = loader.imgs
        train_imgs = train_loader.dataset.imgs
        for i in self.classes:
            if i == 'mole':
                print(i)
            i = i.replace('\n', '')
            data_path_item = self.data_path+i
            image_name = os.listdir(data_path_item)
            for j in image_name:
                if os.path.join(data_path_item+"/",j) not in imgs and os.path.join(data_path_item+"/",j) not in train_imgs:
                    list.append(os.path.join(data_path_item+"/",j))
        return list



def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        # target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(args):
    batch_size = args.batch_size

    train_size = 1000
    test_size = 100

    if args.dataset_type == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cifar_dataset_root = 'E:/datasets/cifar10'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform_train,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform_train)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform_train)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if args.dataset_type == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif args.dataset_type == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif args.dataset_type == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=args.pin_mem,
                                               drop_last=True,
                                               num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=args.pin_mem,
                                              drop_last=True,
                                              num_workers=args.num_workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  pin_memory=args.pin_mem,
                                                  drop_last=True,
                                                  num_workers=args.num_workers)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(args):
    if "cifar" in args.dataset:
        return cifar_dataset(args)

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "Animals_with_Attributes2/classes.txt", "batch_size": args.batch_size},
        "test": {"list_path": args.data_path + "Animals_with_Attributes2/testclasses.txt", "batch_size": args.batch_size}}
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = open(data_config["database"]["list_path"]).readlines()
    classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in classes]
    for data_set in ["train_set", "test"]:
        dsets[data_set] = ImageList(args.data_path,
                                    data_set,
                                    classes,
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

    dsets["database"] = ImageList_database(args.data_path,
                                dsets["test"],
                                classes,
                                open(data_config["test"]["list_path"]).readlines(),
                                transform=transform_train)
    print(data_set, len(dsets["database"]))
    dset_loaders["database"] = util_data.DataLoader(dsets[data_set],
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"],\
           len(dsets["train_set"]), len(dsets["test"]),len(dsets["database"])

def get_train_data(args):
    if "cifar" in args.dataset:
        return cifar_dataset(args)

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "Animals_with_Attributes2/classes.txt", "batch_size": args.batch_size},
        "test": {"list_path": args.data_path + "Animals_with_Attributes2/testclasses.txt", "batch_size": args.batch_size}}
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = open(data_config["database"]["list_path"]).readlines()
    classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in classes]
    # train_classes = open(data_config["train_set"]["list_path"]).readlines()
    # train_classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in train_classes]
    # train_index = [classes.index(i) for i in train_classes]

    for data_set in ["train_set"]:
        dsets[data_set] = ImageList(args.data_path,
                                    data_set,
                                    classes,
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

    # dsets["database"] = ImageList_database(args.data_path,
    #                             dsets["test"],
    #                             dsets["train_set"],
    #                             classes,
    #                             open(data_config["test"]["list_path"]).readlines(),
    #                             transform=transform_train)
    # print("database", len(dsets["database"]))
    # dset_loaders["database"] = util_data.DataLoader(dsets["database"],
    #                                               batch_size=args.batch_size,
    #                                               shuffle=True, num_workers=4)

    return dset_loaders["train_set"], len(dsets["train_set"])

def get_query_data(args, data_loader_train):
    if "cifar" in args.dataset:
        return cifar_dataset(args)

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "Animals_with_Attributes2/classes.txt", "batch_size": args.batch_size},
        "test": {"list_path": args.data_path + "Animals_with_Attributes2/testclasses.txt", "batch_size": args.batch_size}}
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = open(data_config["database"]["list_path"]).readlines()
    classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in classes]
    # train_classes = open(data_config["train_set"]["list_path"]).readlines()
    # train_classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in train_classes]
    # train_index = [classes.index(i) for i in train_classes]

    # with open('/mnt/f88fa63a-2225-40fb-9afa-99a7c125ae28/jy/code/RAZH_AWA2_clu5/classes.txt', "w") as file:
    #     for item in classes:
    #         file.write(str(item) + "\n")

    for data_set in ["test"]:
        dsets[data_set] = ImageList(args.data_path,
                                    data_set,
                                    classes,
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=64,
                                                      shuffle=True, num_workers=10)

    dsets["database"] = ImageList_database(args.data_path,
                                dsets["test"],
                                data_loader_train,
                                classes,
                                open(data_config["test"]["list_path"]).readlines(),
                                transform=transform_train)
    print("database", len(dsets["database"]))
    dset_loaders["database"] = util_data.DataLoader(dsets["database"],
                                                  batch_size=64,
                                                  shuffle=True, num_workers=10)

    return  dset_loaders["test"], dset_loaders["database"],\
           len(dsets["test"]),len(dsets["database"])

def mean_average_precision_R(database_hash, test_hash, database_labels, test_labels, R, num_classes):
    # 设置类别的数量
    # num_classes = args.num_classes

    one_hot_database = np.eye(num_classes)[database_labels.astype(int)]
    one_hot_test = np.eye(num_classes)[test_labels.astype(int)]

    # one_hot_database = database_labels
    # one_hot_test = test_labels

    if R == -1:
        R = database_hash.shape[0]
    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    Recall = []
    # wAPx = []

    for i in tqdm(range(query_num)):  # for i=0
        label = one_hot_test[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch_acg = np.sum(one_hot_database[idx[0:R], :] == label, axis=1)
        imatch = imatch_acg > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)   # 累加
        # Px = Lx.astype(float) / np.arange(1, database_hash.shape[0] + 1, 1)  #
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)

        # L_acg = np.cumsum(imatch_acg)
        # P_acg = L_acg.astype(float) / np.arange(1, R + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
            # wAPx.append(np.sum(imatch * P_acg) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)
            # wAPx.append(0)
        # print(i)

        all_relevant = np.sum(one_hot_database == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    # return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx
    # return np.mean(np.array(APx)), np.mean(np.array(Recall)), np.mean(np.array(wAPx))
    return np.mean(np.array(APx)), np.mean(np.array(Recall))



def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# def get_database_data(test_loader, args):
#     query_image = test_loader.dataset.imgs
#     classes = open(args.data_path + "Animals_with_Attributes2/predicates.txt").readlines()
#     classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in classes]
#
#
#     return 0

import torch
import random
import copy

class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def distance(self, p1, p2):
        return torch.sum((p1 - p2) ** 2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        n = self.data.size(0)
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        set1 = set(old_center)
        set2 = set(new_center)
        return set1 == set2

    def forward(self):
        center = self.generate_center()
        n = self.data.size(0)
        labels = torch.zeros(n).long()
        flag = False
        while not flag:
            old_center = copy.deepcopy(center)

            for i in range(n):
                cur = self.data[i]
                min_dis = 10 * 9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = self.converge(old_center, center)

        return labels, center