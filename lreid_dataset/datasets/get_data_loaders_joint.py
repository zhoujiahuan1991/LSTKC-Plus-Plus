# import torchvision.transforms as T

import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader


import numpy as np

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0,dataset_joint=None):
    # root = osp.join(data_dir, name)
    root = data_dir

    if name is not 'joint':
        dataset = datasets.create(name, root)
        '''select some persons for training'''
        if select_num > 0:
            train = []
            for instance in dataset.train:
                if instance[1] < select_num:
                    # new_id=id_2_id[instance[1]]
                    train.append((instance[0], instance[1], instance[2], instance[3]))

            dataset.train = train
            dataset.num_train_pids = select_num
            dataset.num_train_imgs = len(train)
        '''select some persons for toy datatset'''
        # if select_num>0:
        #     ids=np.linspace(0,dataset.num_train_pids-1,select_num)
        #     ids=np.round(ids).astype('int64')
        #     id_2_id={}
        #     for i in range(len(ids)):
        #         id_2_id[ids[i]]=i
        #     train =[]
        #     for instance in dataset.train:
        #         if instance[1] in ids:
        #             new_id=id_2_id[instance[1]]
        #             train.append((instance[0],new_id,instance[2],instance[3]))

        #     dataset.train=train
        #     dataset.num_train_pids=len(ids)
        #     dataset.num_train_imgs=len(train)
    else:
        dataset = dataset_joint





    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]
def get_test_sample_loader(name,data_dir,batch_size=32,height=256,width=128,num_instances=4):
    # root = osp.join(data_dir, name)
    root = data_dir


    dataset = datasets.create(name, root)

    if 'duke' in name:
        with open('/home/xukunlun/CODE/VoroLReID/xkl_tools/pos.txt', 'r') as f:
            name_list=f.readlines()
            name_list=name_list[:100]
            a=int(name_list[0].replace('\n',''))
            name_list=[int(x.replace('\n','')) for x in name_list]
            new_gallery=[]
            for ins in dataset.gallery:
                if ins[1] in name_list:
                    new_gallery.append(ins)
            dataset.gallery=new_gallery





    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.gallery)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
                   batch_size=batch_size, num_workers=0, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return dataset, num_classes, train_loader, name

def get_joint_data(cfg,all_train_sets,  name='joint_data'):
    all_train_sets=[x[0] for x in all_train_sets]
    # root = osp.join(data_dir, name)
    dataset=copy.deepcopy(all_train_sets[0]) # get an example
    #only consider the datats aboout training
    num_train_pids=dataset.num_train_pids
    num_train_cams=dataset.num_train_cams
    num_train_imgs=dataset.num_train_imgs
    train=dataset.train

    for i in range(1,len(all_train_sets)):
        later_set= all_train_sets[i]

        for instance in later_set.train:
            # change pid and cams
            if later_set.images_dir:
                img_path=os.path.join(later_set.images_dir,instance[0])
            else:
                img_path=instance[0]
            train.append(
                (img_path, instance[1]+num_train_pids,instance[2]+num_train_cams, instance[3])
            )
            # instance[1] += num_train_pids
            # instance[2] += num_train_cams
        num_train_pids += later_set.num_train_pids



        try:
            num_train_cams += later_set.num_train_cams
            num_train_imgs+=later_set.num_train_imgs
        except:
            num_train_imgs += len(later_set.train)

    dataset.num_train_pids=num_train_pids
    dataset.num_train_cams=num_train_cams
    dataset.num_train_imgs=num_train_imgs
    dataset.train=train
    # dataset.images_dir=None


    height, width = (256, 128)
    batch_size=cfg.batch_size
    num_instances=cfg.num_instances
    workers=cfg.workers

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
def get_joint_data(training_loaders):
    dataset_joint=copy.deepcopy(training_loaders[0][0]) # first dataset
    id_count=dataset_joint.num_train_pids
    cam_count=max([x[2] for x in dataset_joint.train])
    for loader in training_loaders[1:]:
        max_cam=0
        dataset, num_classes, train_loader, test_loader, init_loader, name=loader

        for instance in dataset.train:
            dataset_joint.train.append(
                (instance[0],instance[1]+id_count,instance[2]+cam_count,1)
            )
            max_cam=max(max_cam, instance[2])
        id_count+=dataset.num_train_pids
        cam_count+=max_cam+1
    dataset_joint.num_train_pids=id_count


    image_name_list = [os.path.basename(x[0]) for x in dataset_joint.train]
    # image_name_list.sort()
    image_name_list.sort()
    with open('lreid_list_new.txt', 'w') as f:
        for x in image_name_list:
            f.write(x + '\n')
    return dataset_joint




def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, select_num=500) for name in training_set]

    dataset_joint = get_joint_data(training_loaders)
    train_loader_joint = get_data('joint', data_dir, height, width, cfg.batch_size, cfg.workers,
                                  cfg.num_instances, select_num=500, dataset_joint=dataset_joint)

    testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders, train_loader_joint
# def build_data_loaders(cfg,training_set, testing_only_set, toy_num=0):
#     # Create data loaders
#     data_dir=cfg.data_dir
#     height,width=(256,128)
#     training_loaders=[get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
#                  cfg.num_instances, select_num=500) for name in training_set]
#
#     testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
#                                  cfg.num_instances) for name in testing_only_set]
#     return training_loaders, testing_loaders

# def build_data_loaders_toy(cfg,training_set, testing_only_set, select_num):
#     # Create data loaders
#     data_dir=cfg.data_dir
#     height,width=(256,128)
#     training_loaders=[get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
#                  cfg.num_instances, select_num) for name in training_set]
#
#     testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
#                                  cfg.num_instances) for name in testing_only_set]
#     return training_loaders, testing_loaders
