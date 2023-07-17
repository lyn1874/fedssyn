#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fake_data_generate.py
@Time    :   2022/12/13 17:18:14
@Author  :   Bo 
'''
import numpy as np 
import data.prepare_data as prepare_data 
import data.prepare_cifar as prepare_cifar 


def get_fake_data(conf, num_samples, localdata_id):
    print("Add communication round - %d, add %d fake data on each client" % (conf.communication_round, num_samples))
    if conf.dataset == "cifar10" or conf.dataset == "cifar100":
        train_transforms, _ = prepare_cifar.get_cifar_transform(conf)
    if "cifar" in conf.dataset:
        im_group, cls_group = prepare_cifar.get_synthetic_data_from_diffusion(conf)
        print("Selecting local data %d from the entire client collection (%d clients)" % (localdata_id, len(im_group)))
        print(np.unique(cls_group[localdata_id], return_counts=True))
        
    fake_loader = prepare_data.ShapeDsprint(im_group[localdata_id], cls_group[localdata_id], train_transforms)        
    return fake_loader