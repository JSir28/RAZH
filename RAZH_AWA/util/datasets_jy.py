
import os

import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import random
class ImageList(object):

    def __init__(self, data_path, dataset,classes,image_list, transform):
        self.dataset = dataset
        self.data_path = data_path
        self.classes = classes
        self.imgs = self.load_image(image_list, dataset)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        target = [img.split("/")[10].split("_")[0] for img in self.imgs]
        self.target = [self.classes.index(i) for i in target]


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[10].split("_")[0]
        target = self.classes.index(target)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def load_image(self,class_name, dataset):
        list = []
        select_size = 500
        for i in class_name:
            i = i.replace('\n', '')
            data_path_item = self.data_path+i
            image_name = os.listdir(data_path_item)
            if dataset=="train_set":
                if len(image_name)<select_size:
                    image_select = random.sample(image_name, len(image_name))
                    select_size = select_size+(500-len(image_name))
                else:
                    image_select = random.sample(image_name, select_size)
                    select_size = 500

            else:
                image_select = random.sample(image_name,100)
            for j in image_select:
                list.append(os.path.join(data_path_item+"/",j))
        return list
class ImageList_database(object):

    def __init__(self, data_path, test_loader,train_loader,classes, transform):
        self.loader = test_loader
        self.data_path = data_path
        self.classes = classes
        self.imgs = self.load_image(classes, test_loader,train_loader)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[10].split("_")[0]
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
            i = i.replace('\n', '')
            data_path_item = self.data_path+"images/"+i
            image_name = os.listdir(data_path_item)
            for j in image_name:
                if os.path.join(data_path_item+"/",j) not in imgs and os.path.join(data_path_item+"/",j) not in train_imgs:
                    list.append(os.path.join(data_path_item+"/",j))
        return list

class ImageCUBList(object):

    def __init__(self, data_path, dataset,classes,set_class, transform):
        self.dataset = dataset
        self.data_path = data_path
        self.classes = classes
        # image_list = random.sample(self.classes, 150)
        self.train_classes = set_class

        self.imgs = self.load_image(set_class, dataset)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        target = [img.split("/")[-2] for img in self.imgs]
        self.target = [int(i.split(".")[0])-1 for i in target]


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-2].split(".")[0]
        target = int(target)-1

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def load_image(self,class_name, dataset):
        list = []
        select_size = 30
        for i in class_name:
            i = i.replace('\n', '')
            data_path_item = self.data_path+"images/"+i
            image_name = os.listdir(data_path_item)
            if dataset=="train_set":
                if len(image_name)<select_size:
                    image_select = random.sample(image_name, len(image_name))
                    select_size = select_size+(30-len(image_name))
                else:
                    image_select = random.sample(image_name, select_size)
                    select_size = 30

            else:
                image_select = random.sample(image_name,30)
            for j in image_select:
                list.append(os.path.join(data_path_item+"/",j))
        return list
class ImageCUBList_database(object):

    def __init__(self, data_path, test_loader,train_loader,classes, transform):
        self.loader = test_loader
        self.data_path = data_path
        self.classes = classes
        self.imgs = self.load_image(classes, test_loader,train_loader)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-2].split(".")[0]
        target = int(target) - 1

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
            i = i.replace('\n', '')
            data_path_item = self.data_path+"images/"+i
            image_name = os.listdir(data_path_item)
            for j in image_name:
                if os.path.join(data_path_item+"/",j) not in imgs and os.path.join(data_path_item+"/",j) not in train_imgs:
                    list.append(os.path.join(data_path_item+"/",j))
        return list

def get_cub_train_data(args):

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "classes.txt", "batch_size": args.batch_size},
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
        set_class = random.sample(classes, 150)
        dsets[data_set] = ImageCUBList(args.data_path,
                                    data_set,
                                    classes,
                                    set_class,
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=0)
    unseen_classes = [i for i in classes if i not in set_class]
    unseen_classes_index = [classes.index(i) for i in unseen_classes]

    return dset_loaders["train_set"], len(dsets["train_set"]), unseen_classes_index
def get_cub_query_data(args, data_loader_train):

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "classes.txt", "batch_size": args.batch_size},
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
    train_classes = data_loader_train.dataset.train_classes
    for data_set in ["test"]:
        set_classes = [i for i in classes if i not in train_classes]
        dsets[data_set] = ImageCUBList(args.data_path,
                                    data_set,
                                    classes,
                                    set_classes,
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

    dsets["database"] = ImageCUBList_database(args.data_path,
                                dsets["test"],
                                data_loader_train,
                                classes,
                                transform=transform_train)
    print("database", len(dsets["database"]))
    dset_loaders["database"] = util_data.DataLoader(dsets["database"],
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)

    return  dset_loaders["test"], dset_loaders["database"],\
           len(dsets["test"]),len(dsets["database"])


class ImageCIFAR10List(object):

    def __init__(self, data_path, dataset,classes,set_class, transform):
        self.dataset = dataset
        self.data_path = data_path
        self.classes = classes
        # image_list = random.sample(self.classes, 150)
        self.train_classes = set_class

        self.imgs = self.load_image(set_class, dataset)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        # target = [img.split("/")[-2] for img in self.imgs]
        # self.target = [classes.index(i) for i in target]


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-2]
        target = self.classes.index(target)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def load_image(self,class_name, dataset):
        list = []
        if dataset=="train_set":
            select_size = 500
        else:
            select_size = 100
        for i in class_name:
            i = i.replace('\n', '')
            data_path_item = self.data_path+i
            image_name = os.listdir(data_path_item)
            if dataset=="train_set":
                if len(image_name)<select_size:
                    image_select = random.sample(image_name, len(image_name))
                    select_size = select_size+(500-len(image_name))
                else:
                    image_select = random.sample(image_name, select_size)
                    select_size = 500

            else:
                image_select = random.sample(image_name,100)
            for j in image_select:
                list.append(os.path.join(data_path_item+"/",j))
        return list
class ImageCIFAR10_database(object):

    def __init__(self, data_path, test_loader,train_loader,classes, transform):
        self.loader = test_loader
        self.data_path = data_path
        self.classes = classes
        self.imgs = self.load_image(classes, test_loader,train_loader)
            # [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform


    def __getitem__(self, index):
        path = self.imgs[index]
        target = path.split("/")[-2]
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
            i = i.replace('\n', '')
            data_path_item = self.data_path+i
            image_name = os.listdir(data_path_item)
            for j in image_name:
                if os.path.join(data_path_item+"/",j) not in imgs and os.path.join(data_path_item+"/",j) not in train_imgs:
                    list.append(os.path.join(data_path_item+"/",j))
        return list
def get_cifar10_train_data(args):

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "classes.txt", "batch_size": args.batch_size},
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
        set_class = random.sample(classes, 8)
        dsets[data_set] = ImageCIFAR10List(args.data_path,
                                    data_set,
                                    classes,
                                    set_class,
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=0)
    unseen_classes = [classes.index(i) for i in classes if i not in set_class]

    return dset_loaders["train_set"], len(dsets["train_set"]), unseen_classes
def get_cifar10_query_data(args, data_loader_train):

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": args.data_path + "Animals_with_Attributes2/trainclasses.txt", "batch_size": args.batch_size},
        "database": {"list_path":args.data_path + "classes.txt", "batch_size": args.batch_size},
        "test": {"list_path": args.data_path + "Animals_with_Attributes2/testclasses.txt", "batch_size": args.batch_size}}
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = open(data_config["database"]["list_path"]).readlines()
    classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in classes]
    # train_classes = open(data_config["train_set"]["list_path"]).readlines()
    # train_classes = [i.replace('\t', ' ').replace('\n', '').split(' ')[-1] for i in train_classes]
    # train_index = [classes.index(i) for i in train_classes]
    train_classes = data_loader_train.dataset.train_classes
    for data_set in ["test"]:

        set_classes = [i for i in classes if i not in train_classes]
        dsets[data_set] = ImageCIFAR10List(args.data_path,
                                    data_set,
                                    classes,
                                    set_classes,
                                    transform=transform_train)
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)

    dsets["database"] = ImageCIFAR10_database(args.data_path,
                                dsets["test"],
                                data_loader_train,
                                classes,
                                transform=transform_train)
    print("database", len(dsets["database"]))
    dset_loaders["database"] = util_data.DataLoader(dsets["database"],
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)

    return  dset_loaders["test"], dset_loaders["database"],\
           len(dsets["test"]),len(dsets["database"])

