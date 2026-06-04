#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, us_b_pairs, us_b_graph, num_bundles, us_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        # self.u_b_pairs = u_b_pairs
        # self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample
        self.us_b_pairs = us_b_pairs
        self.us_b_graph = us_b_graph

        self.u_b_for_neg_sample = us_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_set_b, pos_bundle = self.us_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.us_b_graph[user_set_b, i] == 0 and not i in all_bundles:
                all_bundles.append(i)
                if len(all_bundles) == self.neg_sample+1:           # 得到负例
                    break                                                                                                               

        return torch.LongTensor([user_set_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.us_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, us_b_pairs, us_b_graph, us_b_graph_train, num_users_set, num_bundles):
        self.us_b_pairs = us_b_pairs
        self.us_b_graph = us_b_graph
        self.train_mask_us_b = us_b_graph_train

        self.num_users_set = num_users_set
        self.num_bundles = num_bundles
        indice_set = np.array(us_b_pairs, dtype=np.int32)
        self.users_set = torch.arange(num_users_set, dtype=torch.long).unsqueeze(dim=1)
        # self.bundles = torch.arange(num_bundles, dtype=torch.long)
        self.bundles = torch.tensor(indice_set[:, 1], dtype=torch.long)

    def __getitem__(self, index):
        us_b_grd = torch.from_numpy(self.us_b_graph[index].toarray()).squeeze()
        us_b_mask = torch.from_numpy(self.train_mask_us_b[index].toarray()).squeeze()

        return index, us_b_grd, us_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items, self.num_users_set = self.get_data_size()
        self.item_weights = np.zeros((self.num_items, 1), dtype=float)
        self.get_item_weights()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")
        # set
        us_b_pairs_train, us_b_graph_train = self.get_us_b("train")
        us_b_pairs_val, us_b_graph_val = self.get_us_b("tune")
        us_b_pairs_test, us_b_graph_test = self.get_us_b("test")

        us_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, us_b_pairs_train, us_b_graph_train, self.num_bundles, us_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(us_b_pairs_val, us_b_graph_val, us_b_graph_train, self.num_users_set, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(us_b_pairs_test, us_b_graph_test, us_b_graph_train, self.num_users_set, self.num_bundles)

        self.graphs = [us_b_graph_train, u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=0)

        self.bundles_index_list_train, self.herbs_id_list_train = self.get_test_b_index('train')
        self.bundles_index_list_tune, self.herbs_id_list_tune = self.get_test_b_index('tune')
        self.bundles_index_list_test, self.herbs_id_list_test = self.get_test_b_index('test')


    def get_data_size(self):
        name = self.name
        # if "_" in name:
        #     name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split(' ')][:4]


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        # 构建B-I graph --> 压缩矩阵 csr_matrix - shape(bundle num, item num)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        # 构建U-I graph --> 压缩矩阵 csr_matrix - shape(user num, item num)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph

    def get_item_weights(self):
        with open(os.path.join(self.path, self.name, 'user_item_train.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        # 统计u_i_pairs中，每个item出现的次数
        for u, i in u_i_pairs:
            self.item_weights[i][0] += 1
        print('item_weight ', len(self.item_weights))
        item_freq_max = self.item_weights.max()
        print('item_freq: item_weight.shape[0] and [1]', self.item_weights.shape[0], ' ',
              self.item_weights.shape[1])
        for index in range(self.item_weights.shape[0]):
            self.item_weights[index][0] = item_freq_max * 1.0 / self.item_weights[index][0]

    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        # 构建U-B graph --> 压缩矩阵 csr_matrix - shape(user num, bundle num)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph

    def get_us_b(self, task):
        with open(os.path.join(self.path, self.name, 'set_user_bundle_{}.txt'.format(task)), 'r') as f:
            us_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(us_b_pairs, dtype=np.int32)
        values = np.ones(len(us_b_pairs), dtype=np.float32)
        # 构建U-B graph --> 压缩矩阵 csr_matrix - shape(user num, bundle num)
        us_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users_set, self.num_bundles)).tocsr()

        print_statistics(us_b_graph, "US-B statistics in %s" %(task))

        return us_b_pairs, us_b_graph

    def get_test_b_index(self, task):
        prescription_list = list()
        with open(os.path.join(self.path, self.name, 'set_user_bundle_{}.txt'.format(task)), 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempH = int(temp[1])
                    prescription_list.append(tempH)
        herbs_id_list = list()
        with open(os.path.join(self.path, self.name, '{}_id.txt'.format(task)), 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    temp = l.strip().split('\t')
                    tempH = temp[1].split(' ')
                    herbs_id_list.append(tempH)
        return prescription_list, herbs_id_list

