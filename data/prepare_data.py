#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_data.py
@Time    :   2022/11/30 13:08:14
@Author  :   Bo 
'''
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import cv2 
import torch 

    
class ShapeDsprint(Dataset):
    def __init__(self, image, label, transform):
        super().__init__()
        self.image = image
        self.targets = label
        self.transform = transform
        self.index = np.arange(len(self.image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        if self.transform is not None:
            image = self.transform(image)
        s_label = self.targets[index]
        return image, s_label 
    

def get_dataloader(tr_data, val_data, batch_size, workers):
    train_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            drop_last=True, pin_memory=True, num_workers=workers)
    return train_loader, val_loader


def get_test_tensor(image, image_shape=[32, 32, 3], device=torch.device("cuda")):
    images = np.concatenate([[cv2.imread(v).astype(np.float32)] for v in image], axis=0)[:, :, :, ::-1]
    images = images / 255.0    
    image_npy = images.copy()
    images = torch.from_numpy(images).to(torch.float32).permute(0, 3, 1, 2)
    return images.to(device).view(-1, image_shape[2], image_shape[0], image_shape[1]), image_npy 


def get_test_tensor_npy(image_loader, batch_size, shuffle):
    val_loader = DataLoader(image_loader, batch_size=batch_size, shuffle=shuffle,
                            drop_last=False, pin_memory=True, num_workers=1)
    return val_loader 



