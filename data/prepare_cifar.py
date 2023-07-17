#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_data.py
@Time    :   2022/09/12 09:19:20
@Author  :   Bo 
'''
import os
import numpy as np 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch 
import utils.utils as utils 


def _get_cifar(conf, name, root, split, transform, target_transform, download):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    root: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    is_train = True if "train" in split else False

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )
        
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        if conf.apply_transform:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    
    
def get_cifar_transform(conf):
    if conf.dataset == "cifar10":
        normalize = (
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
    elif conf.dataset == "cifar100":
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )
        
    normalize = normalize if conf.pn_normalize == True else None 
    tr_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),             
                ]
                + ([normalize] if normalize is not None else [])
            )
    tt_transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    return tr_transform, tt_transform
    
    
def get_dataset(conf, name, datasets_path, split="train", transform=None, target_transform=None,
                download=True):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    datasets_path: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)
    if name == "cifar10" or name == "cifar100":
        return _get_cifar(
            conf, name, root, split, transform, target_transform, download
            )


def get_synthetic_data_from_diffusion(conf):
    if conf.dataset == "cifar10":
        if "add_fake_diffusion_sync_local_generator" in conf.align_data:
            im_group, label_group = combine_synthetic_data_from_local_generators(conf)
    elif conf.dataset == "cifar100":
        if "add_fake_diffusion_sync_local_generator" in conf.align_data:
            im_group, label_group = combine_synthetic_data_from_local_generators(conf)
    if "without_shuffle" not in conf.align_data:
        data_index = np.arange(len(im_group))
        split_index = np.split(data_index, conf.n_clients)
        
        path2save_index = "%s/%s/%s/synthetic_sampling_index" % (utils.get_path_init("home"), conf.folder_name, conf.dir_name)
        if conf.communication_round == 0 and os.path.isfile(path2save_index + ".npy") == False:
            shuffle_index = np.random.choice(data_index, len(data_index), replace=False)
            np.save(path2save_index, shuffle_index)
        else:
            shuffle_index = np.load(path2save_index + ".npy", allow_pickle=True) 
            print("Load shuffle index")       
        im_group = im_group[shuffle_index]
        label_group = label_group[shuffle_index]        
        
        im_per_client = [im_group[v] for v in split_index]
        cls_per_client = [label_group[v] for v in split_index]

    elif "without_shuffle" in conf.align_data:
        im_per_client = im_group 
        cls_per_client = label_group         
    im_per_client = [(v / 255.0).astype(np.float32) for v in im_per_client]
    print([np.shape(v) for v in im_per_client])
    print([np.max(v) for v in im_per_client])
    print([np.shape(v) for v in cls_per_client])
    return im_per_client, cls_per_client 


def combine_synthetic_data_from_local_generators(conf, save=False, num_devices=10):
    data_dir = conf.synthetic_path + "/version_%02d_dataset_%s_non_iid_alpha_%.2f_num_selection_%d/" % (conf.synthetic_version, 
                                                                                                        conf.dataset, 
                                                                                                        conf.non_iid_alpha, 
                                                                                                        conf.num_images_train_synthetic)
    im_group, label_group = [], []
    for i in range(conf.n_clients):
        sub_folder = data_dir + "/client_id_%02d/" % i 
        if conf.synthetic_epoch == 500:
            im_path = sub_folder + "UNet_%s-250-sampling_steps-%d_images-class_condn_True.npz" % (conf.dataset, conf.num_synthetic_images)
        else:
            im_path = sub_folder + "UNet_%s_%d-250-sampling_steps-%d_images-class_condn_True.npz" % (conf.dataset, conf.synthetic_epoch, 
                                                                                                     conf.num_synthetic_images)
        if not os.path.isfile(im_path):
            select_subset_data = True 
            num_images = 5000 if conf.n_clients == 10 else 1125 
            if conf.synthetic_epoch == 500:
                im_path = sub_folder + "UNet_%s-250-sampling_steps-%d_images-class_condn_True.npz" % (conf.dataset, num_images)
            else:
                im_path = sub_folder + "UNet_%s_%d-250-sampling_steps-%d_images-class_condn_True.npz" % (conf.dataset, conf.synthetic_epoch, 
                                                                                                         num_images)
        else:
            select_subset_data = False 
            num_images = conf.num_synthetic_images
        data = np.load(im_path)
        if not select_subset_data:
            im_group.append(data["arr_0"])
            label_group.append(data["arr_1"])
        else:
            if conf.num_synthetic_images < num_images:
                cls_, cls_freq = np.unique(data["arr_1"], return_counts=True)
                cls_num = [int(np.ceil(conf.num_synthetic_images * (v / np.sum(cls_freq)))) for v in cls_freq]
                sub_set_index = [np.where(data["arr_1"] == v)[0][:cls_num[i]] for i, v in enumerate(cls_)]
                sub_set_index = np.concatenate(sub_set_index, axis=0)[:conf.num_synthetic_images]
                im_group.append(data["arr_0"][sub_set_index])
                label_group.append(data["arr_1"][sub_set_index])
            else:
                im_group.append(np.concatenate([data["arr_0"], data["arr_0"]], axis=0))
                label_group.append(np.concatenate([data["arr_1"], data["arr_1"]], axis=0))
        # print("Client %02d: " % i, " class distribution: ", np.unique(label_group[-1], return_counts=True))
        
    if "without_shuffle" not in conf.align_data:
        im_group = np.concatenate(im_group, axis=0)
        label_group = np.concatenate(label_group, axis=0)
    
        print("===========Status of the class distribution in the synthetic dataset===============")
        print(np.unique(label_group, return_counts=True))
        print("===================================================================================")
    return im_group, label_group 



def aggregate_synthetic_together(synthetic_version, non_iid_alpha,
                                 num_images_train_synthetic, num_synthetic_images, 
                                 num_devices=10, dataset="cifar10", 
                                 save=False):
    class PARAM:
        dataset = "cifar10"
    conf = PARAM 
    conf.synthetic_version = synthetic_version
    conf.dataset = dataset 
    conf.non_iid_alpha = non_iid_alpha
    conf.num_images_train_synthetic = num_images_train_synthetic
    conf.n_clients = num_devices
    conf.dataset = dataset 
    conf.num_synthetic_images = num_synthetic_images
    output = combine_synthetic_data_from_local_generators(conf, save, num_devices)
    print("Done aggregating all the synthetic images")
    