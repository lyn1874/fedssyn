#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fed_dyn.py
@Time    :   2023/01/25 09:22:53
@Author  :   Bo 
'''

import torch 
import torch.nn as nn 
import os 
import numpy as np
import pickle 
import fed_model.cls_model as cls_model 
import data.create_dataset as create_dataset
import utils.utils as utils 


device=torch.device("cuda") 


class Aggregator(object):
    def __init__(self, conf, path, select_client=[], communication_round=0, 
                 evaluate_on_tt=False, device=torch.device("cuda")):
        self.conf = conf 
        self.path = path 
        self.n_clients = conf.n_clients 
        self.communication_round = communication_round
        self.path_use = self.path + "/communication_round_%03d/" % self.communication_round
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
        rep = utils.get_replace_for_init_path(conf.loc)

        path2save = self.path.replace(rep, "../") + "statistics.obj"
        if not os.path.exists(path2save.replace("/statistics.obj", "")):
            os.makedirs(path2save.replace("statistics.obj", ""))
        if communication_round == 0:
            self.stat = {}
            for k in range(self.n_clients):
                self.stat["client_%02d_accuracy" % (k)] = []
                self.stat["client_%02d_loss" % k] = []
            self.stat["aggregate_accuracy"] = []
            self.stat["aggregate_loss"] = []  
        else:          
            self.stat = pickle.load(open(path2save, "rb"))
        
        if len(select_client) == 0:
            select_client = np.arange(self.n_clients)
        self.select_client = select_client
        self.ratio = torch.FloatTensor([1.0 / len(self.select_client) for _ in self.select_client]).to(device)
        self.full_ratio = torch.FloatTensor([1.0 / conf.n_clients for _ in self.select_client]).to(device)
        
        self.tt_data_loader = self._get_data()
        self.ckpt_dir = self._get_ckpt_dir()
        self.weights_group = self._get_model(evaluate_on_tt)
        delta_c = self._load_delta_hidden()
        hidden_stat = self._get_hidden_state(delta_c)
        self._get_global_model(self.weights_group, hidden_stat)
        print("-------------------save model----------------------")
        with open(path2save, "wb") as f:
            pickle.dump(self.stat, f)
            
    def _convert_tensor_to_same_type(self, tt_convert):
        for k in tt_convert.keys():
            tt_convert[k] = tt_convert[k].to(device)    
        return tt_convert 

    def _load_delta_hidden(self):
        sub_path = sorted([self.path_use + v for v in os.listdir(self.path_use) if "client_id" in v and int(v.split("_")[-1]) in self.select_client])
        assert len(sub_path) == len(self.select_client)
        delta_c = {}        
        for i, s_path in enumerate(sub_path):
            _delta_dir = s_path + "/client_hidden_delta_%02d.obj" % self.select_client[i]
            delta_ci = pickle.load(open(_delta_dir, "rb"))
            delta_ci = self._convert_tensor_to_same_type(delta_ci)
            if i == 0:
                for k in delta_ci.keys():
                    delta_c[k] = delta_ci[k] * self.full_ratio[i]
            else:
                for k in delta_ci.keys():
                    delta_c[k] += (delta_ci[k] * self.full_ratio[i])
        return delta_c
    
    def _get_hidden_state(self, delta_hidden):
        if self.communication_round > 0:
            previous_c_path = self.path + "/communication_round_%03d/" % (self.communication_round - 1)
        if self.communication_round == 0:
            hidden_stat = {}
            for k in delta_hidden.keys():
                hidden_stat[k] = 0 - delta_hidden[k] * self.conf.beta 
        else:
            hidden_previous = pickle.load(open(previous_c_path + "/hidden_stat.obj", "rb"))
            hidden_previous = self._convert_tensor_to_same_type(hidden_previous)
            hidden_stat = {}
            for k in hidden_previous.keys():
                hidden_stat[k] = hidden_previous[k] - self.conf.beta * delta_hidden[k]
        with open(self.path_use + "/hidden_stat.obj", "wb") as f:
            pickle.dump(hidden_stat, f)             
        return hidden_stat 
    
    def _get_global_model(self, weights_group, hidden_stat):
        weights_update = {}
        for k in hidden_stat.keys():
            weights_update[k] = weights_group[k] - 1 / self.conf.beta * hidden_stat[k]
        torch.save(weights_update, self.path_use + "/aggregated_model.pt")

    def _get_ckpt_dir(self):
        sub_path = sorted([self.path_use + v for v in os.listdir(self.path_use) if "client_id" in v and int(v.split("_")[-1]) in self.select_client])
        assert len(sub_path) == len(self.select_client)
        ckpt_dir = []
        for i, s_path in enumerate(sub_path):
            all_ckpt = [s_path + "/" + v for v in os.listdir(s_path) if ".pt" in v]
            accu = [int(v.split("-")[1]) for v in all_ckpt]
            ckpt_dir.append(all_ckpt[np.argmax(accu)])
            print(all_ckpt[np.argmax(accu)])
        return ckpt_dir 
    
    def _get_model(self, evaluate=False):
        weights_group = {}
        for i, s_ckpt in enumerate(self.ckpt_dir):
            model_value = torch.load(s_ckpt, map_location=device)
            if evaluate:
                _, _, accu, loss = self._evaluate_model(model_value)
                self.stat["client_%02d_accuracy" % self.select_client[i]].append(accu.detach().cpu().numpy())
                self.stat["client_%02d_loss" % self.select_client[i]].append(loss.detach().cpu().numpy())
            if i == 0:
                for k in model_value.keys():
                    weights_group[k] = model_value[k] * self.ratio[i]
            else:
                for k in model_value.keys():
                    weights_group[k] += model_value[k] * self.ratio[i]
            del model_value 
        pred, tt_label, accu, loss = self._evaluate_model(weights_group)        
        self.stat["aggregate_accuracy"].append(accu.detach().cpu().numpy())
        self.stat["aggregate_loss"].append(loss.detach().cpu().numpy())
        return weights_group
    
    def _get_data(self):
        _, tt_data_loader = create_dataset.get_test_dataset(self.conf, shuffle=False)
        return tt_data_loader 
    
    def _evaluate_model(self, model_param):
        model_use = cls_model.get_model(self.conf, device)
        model_use.eval()
        model_use.load_state_dict(model_param)
        model_use.requires_grad_(False)
        model_use.to(device)
        accu = 0.0 
        num_data = 0.0
        tt_loss = 0.0
        pred, tt_label = [], []
        for i, s_data in enumerate(self.tt_data_loader):
            _x, _y = s_data[0].to(device), s_data[1].to(device)
            _pred = model_use(_x)
            tt_loss += self.loss_func(_pred, _y)
            accu += (_pred.argmax(axis=-1) == _y).sum()
            num_data += len(_x)
            pred.append(_pred.argmax(axis=-1).detach().cpu().numpy())
            tt_label.append(_y.detach().cpu().numpy())
        return np.concatenate(pred), np.concatenate(tt_label), accu / num_data, tt_loss / num_data
        
        
