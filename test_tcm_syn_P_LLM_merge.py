#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from utility_ywj import Datasets
import pickle
import numpy as np
from helper import *
import sys
import wandb
from config_weight_test import CONFIG
import pandas as pd
import random


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Herb_BGCN", type=str, help="which dataset to use, options: NetEase, Youshu, iFashion")
    parser.add_argument("-m", "--model", default="KCCL", type=str, help="which model to use, options: KCCL")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-c", "--config", default="CONFIG_reg:7e-3-pair-MD-best", type=str,
                        help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    return args

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def main():
    # conf = yaml.safe_load(open("./config.yaml"))
    # server_model_save_path = "/mnt/local/wjyue/CorssCBR/model/visual"
    server_model_save_path = "./visual/SYN_P_merge"
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] in ["KCCL"], "Pls select models from: KCCL"

    # if "_" in dataset_name:
    #     conf = conf[dataset_name.split("_")[0]]
    # else:
    #     conf = conf[dataset_name]
    CONFIG_name = paras["config"]
    conf = CONFIG[CONFIG_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    conf["num_users_set"] = dataset.num_users_set

    # os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(conf["gpu"])
    device = torch.device("cuda:" + str(conf["gpu"]) if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)
    print(torch.cuda.device_count())
    us_u = load_obj('./datasets/Herb_BGCN/us_u_list').to(device)   # [4560, 360]

    result_path = f"{server_model_save_path}/result/%s/%s" % (conf["dataset"], conf["model"])
    file_path = result_path + f"/result_test-{CONFIG_name}-LLM_merge-seed.xlsx"
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.exists(file_path):
        # 创建一个DataFrame
        columns = ["Parameter", "Recall@5",
                   "Recall@10", "Recall@15",
                   "Recall@20",
                   "Pre@5",
                   "Pre@10", "Pre@15",
                   "Pre@20",
                   "NDCG@5",
                   "NDCG@10", "NDCG@15",
                   "NDCG@20",
                   "RMRR@5",
                   "RMRR@10", "RMRR@15",
                   "RMRR@20"]
        df = pd.DataFrame(columns=columns)
        df.to_excel(file_path, index=False)
    # 读取现有的Excel文件
    df = pd.read_excel(file_path)

    for lr, l2_reg, item_level_ratio, bundle_level_ratio, bundle_agg_ratio, embedding_size, num_layers, c_lambda, c_temp, LLM_name, seed in \
            product(conf['lrs'], conf['l2_regs'], conf['item_level_ratios'], conf['bundle_level_ratios'], conf['bundle_agg_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"], conf["LLM_name"], conf['seed']):
        run_path = f"{server_model_save_path}/runs/%s/%s" %(conf["dataset"], conf["model"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if not os.path.isdir(run_path):
            os.makedirs(run_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert item_level_ratio == 0 and bundle_level_ratio == 0 and bundle_agg_ratio == 0

        settings += ["Neg_%d" %(conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)]

        conf["item_level_ratio"] = item_level_ratio
        conf["bundle_level_ratio"] = bundle_level_ratio
        conf["bundle_agg_ratio"] = bundle_agg_ratio
        conf["num_layers"] = num_layers
        settings += [str(item_level_ratio), str(bundle_level_ratio), str(bundle_agg_ratio), str(num_layers)]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp), str(num_layers)]
        setting = "_".join(settings)

        print("args.pretrain\t", conf['pretrain'])
        if conf['pretrain'] == 1:
            print("pretrain==1")
            weights_save_path = os.path.join(server_model_save_path, conf["dataset"],
                                             conf['model'], LLM_name,
                                             f"lr{lr}_reg{l2_reg}_aug{conf['aug_type']}"
                                             f"_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}"
                                             f"_ctemp-{c_temp}_numlayer-{num_layers}_{conf['save_tail']}_merge-{seed}/")
            pretrain_path = weights_save_path
            print('load the pretrained model parameters from: ', pretrain_path)
            model = torch.load(weights_save_path + 'model.pkl', map_location=lambda storage, loc: storage)
            if model:
                print("start to load pretrained model")
                model = model.to(conf["device"])
                model.conf['device'] = device
                print("model after add device:", model)
                paras_test = f"{LLM_name}_merge-lr{lr}_reg{l2_reg}_ctemp-{c_temp}_aug-{conf['aug_type']}_numlayer-{num_layers}_emb-{str(conf['embedding_sizes'])}_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}_{str(seed)}"
                ret = test(model, conf, dataset, us_u)  # 测试使用集合列表
                pretrain_ret = "pretrained model \trecall=[%s], precision=[%s], ndcg=[%s], rmrr=[%s]" % \
                               ('\t'.join(['%.5f' % r for r in ret['recall']]),
                                '\t'.join(['%.5f' % r for r in ret['precision']]),
                                '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                                '\t'.join(['%.5f' % r for r in ret['rmrr']]),
                                )
                print(pretrain_ret)
                print("end:")
                # 写入文件
                new_df = pd.DataFrame([[paras_test,
                                        round(ret["recall"][0], 5),
                                        round(ret["recall"][1], 5), round(ret["recall"][2], 5),
                                        round(ret["recall"][3], 5),
                                        round(ret["precision"][0], 5),
                                        round(ret["precision"][1], 5), round(ret["precision"][2], 5),
                                        round(ret["precision"][3], 5),
                                        round(ret["ndcg"][0], 5),
                                        round(ret["ndcg"][1], 5), round(ret["ndcg"][2], 5),
                                        round(ret["ndcg"][3], 5),
                                        round(ret["rmrr"][0], 5), round(ret["rmrr"][1], 5), round(ret["rmrr"][2], 5),
                                        round(ret["rmrr"][3], 5)]],
                                      columns=df.columns)
                df = pd.concat([df, new_df], ignore_index=True)

                # 写入Excel文件
                df.to_excel(file_path, index=False)
                print(f"New results have been added to {file_path}")
            else:
                print('no model, without pretraining.')

        else:
            print('no pretrain, without pretraining.')

# def range_test(model, bundle_val_data, conf, us_u, bundles_index_list_tune):
#     set_K = conf["set_K"]
#     test_us_b_pairs = np.array(bundle_val_data.us_b_pairs, dtype=np.int32)
#     users_to_test, ground_truth_us_b = test_us_b_pairs[:, 0], test_us_b_pairs[:, 1]
#     test_users = torch.tensor(users_to_test, dtype=torch.long).to(conf["device"])  # [TN, ]
#     ground_truth_us_b = torch.tensor(ground_truth_us_b, dtype=torch.long).to(conf["device"])
#     result = {'precision': np.zeros(len(set_K)), 'recall': np.zeros(len(set_K)), 'mrr': 0}
#     model.eval()
#     with torch.no_grad():
#         rs = model.propagate(test=True)
#         pred_b = model.evaluate(rs, test_users, us_u, bundle_val_data.bundles.to(conf["device"]))
#         values, indices_index = torch.sort(pred_b, dim=1, descending=True)  # [n_test, n_bundle]
#         bundles_index_list_tune = torch.tensor(bundles_index_list_tune).to(conf["device"])
#         indices = [bundles_index_list_tune[i] for i in indices_index]
#         mrr_score = 0.
#         recall_list = np.zeros(len(set_K))  # [1, 5, 10, 15, 20]
#         pre_list = np.zeros(len(set_K))
#         test_index = 0
#         for set_id in users_to_test:
#             score = 0.
#             posit_index = bundle_val_data.us_b_graph[set_id].indices  # 正例下标列表
#             for pi in posit_index:
#                 range_index_all = (indices[test_index] == int(pi)).nonzero().squeeze(dim=1).to(conf["device"])
#                 range_index = range_index_all[0]  # 正例下标在排序中的位置
#                 mrr = torch.reciprocal(range_index + 1)
#                 # mrr = mrr.item()     # 计算mrr值
#                 x1 = posit_index.size
#                 if mrr >= 1 / posit_index.size:  # 多个正例（M)的情况下，只要推荐的结果在前M位上就认为是推荐在了第一位上。
#                     mrr = 1.
#                 score += mrr
#             score = score / posit_index.size
#             mrr_score += score
#             # 计算Recall值
#             for ii in range(len(set_K)):
#                 k = set_K[ii]  # 推荐的药方的个数 1、5、10、15、20
#                 rec_id = indices[test_index][:k]  # 推荐的K个药方id
#                 number = 0
#                 for pi in posit_index:  # 真实药方id
#                     if pi in rec_id:
#                         number += 1
#                 recall_list[ii] = recall_list[ii] + number / posit_index.size
#                 pre_list[ii] = pre_list[ii] + number / k
#             test_index += 1
#         mrr_score = mrr_score / len(users_to_test)  # 平均
#         result['mrr'] = mrr_score.item()
#         for ii in range(len(set_K)):
#             result['recall'][ii] = recall_list[ii] / len(users_to_test)  # 平均
#             result['precision'][ii] = pre_list[ii] / len(users_to_test)
#         return result


def test(model, conf, dataset, us_u):
    """
    Args:
    """
    Ks = conf["set_K"]
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # test_users = users_to_test
    bundle_test_data = dataset.bundle_test_data
    test_us_b_pairs = np.array(bundle_test_data.us_b_pairs, dtype=np.int32)
    users_to_test, ground_truth_us_b = test_us_b_pairs[:, 0], test_us_b_pairs[:, 1]
    test_users = torch.tensor(users_to_test, dtype=torch.long).to(conf["device"])  # [TN, ]
    model.eval()
    with torch.no_grad():
        rs = model.propagate(test=True)
        positive_bundles = ground_truth_us_b
        # 由（512,1）的Tensor转为列表
        positive_bundles = positive_bundles.tolist()
        # positive_bundles中第一维的每个值是self.bi_graph中的行号, 通过行号取出值，即正例的item
        _, _, _, b_i_graph = dataset.graphs
        rate_batch = model.evaluate_items(rs, test_users, us_u)
        print('rate_batch ', rate_batch.shape)

        values, indices_index = torch.sort(rate_batch, dim=1, descending=True)  # [n_test, all_items]
        item_batch = range(conf["num_items"])

        precision_n = np.zeros(len(Ks))
        recall_n = np.zeros(len(Ks))
        ndcg_n = np.zeros(len(Ks))
        rmrr_n = np.zeros(len(Ks))
        topN = Ks

        gt_count = 0
        candidate_count = 0
        for index in range(len(positive_bundles)):
            entry = [us_index for us_index, us_id in enumerate(users_to_test) if us_id==users_to_test[index]]
            # entry = positive_bundles[index]
            v_list = []
            for en_i in entry:
                v = dataset.herbs_id_list_test[en_i] # sym-index's true herb set list
                v_list.append(v)
            rating = indices_index[index]
            candidate_count += len(rating)
            # rating.sort(key=lambda x: x[1], reverse=True)
            K_max = topN[len(topN) - 1]
            for ii in range(len(topN)):  # topN: [5, 10, 15, 20]
                top_recall, top_precision, top_ndcg, top_rmrr, top_iou = 0., 0., 0., 0., 0.
                for v in v_list:  # v:对应的ground truth
                    r = []
                    for i in rating[:K_max]:
                        herb = str(i.item())
                        if herb in v:
                            r.append(1)
                        else:
                            r.append(0)
                    number = 0
                    herb_results = []  # 推荐列表中herb 集合
                    for i in rating[:topN[ii]]:
                        herb = str(i.item())
                        herb_results.append(herb)
                        if herb in v:
                            number += 1
                    # todo: modified MRR to Rank-MRR
                    mrr_score = 0.
                    for a_rank in range(len(v)):  # herb 在grand truth中的位置a_rank
                        if v[a_rank] in herb_results:
                            a_refer = herb_results.index(v[a_rank])  # herb 在推荐列表中的位置a_refer
                            mrr_score += 1.0 / (abs(a_refer - a_rank) + 1)
                    if float(number / topN[ii]) > top_precision:  # 使用precision选择GT
                        top_precision = float(number / topN[ii])
                        top_recall = float(number / len(v))
                        top_ndcg = ndcg_at_k(r, topN[ii])
                        top_rmrr = mrr_score / len(v)
                precision_n[ii] = precision_n[ii] + top_precision  # [ii]所有测试数据top k的precision之和
                recall_n[ii] = recall_n[ii] + top_recall
                ndcg_n[ii] = ndcg_n[ii] + top_ndcg
                rmrr_n[ii] = rmrr_n[ii] + top_rmrr
        print('gt_count ', gt_count)
        print('candidate_count ', candidate_count)
        print('ideal candidate count ', len(positive_bundles) * conf["num_items"])
        for ii in range(len(topN)):
            result['precision'][ii] = precision_n[ii] / len(positive_bundles)
            result['recall'][ii] = recall_n[ii] / len(positive_bundles)
            result['ndcg'][ii] = ndcg_n[ii] / len(positive_bundles)
            result['rmrr'][ii] = rmrr_n[ii] / len(positive_bundles)
    return result

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    # dcg_max = dcg_at_k(np.ones_like(r), k, method)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


if __name__ == "__main__":
    main()
