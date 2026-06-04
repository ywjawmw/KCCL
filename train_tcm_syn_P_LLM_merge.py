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
from models.KCCL_syn_P_LLM_merge import KCCL
import pickle
import numpy as np
from helper import *
import sys
import wandb
from config_weight import CONFIG
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from models.AutoLoss import AutomaticWeightedLoss
import torch.nn as nn
import time

# 纯0不考虑

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

def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def first_digit(num):
    """
    获取一个数字的第一位数字。
    """
    # 处理负数
    num = abs(num)

    # 如果数字是 0，返回 0
    if num == 0:
        return 0

    # 获取科学计数法表示
    while num >= 10:
        num /= 10
    while num < 1:
        num *= 10

    return int(num)


def same_order_of_magnitude(a, b):
    """
    判断两个数字是否属于同一个数量级。
    """
    exp_a = int(math.floor(math.log10(abs(a))))
    exp_b = int(math.floor(math.log10(abs(b))))
    return exp_a == exp_b, 10**(exp_a)


def loss_magnitude(loss1, loss2):
    """
    判断两个损失值的数量级, 如果两个值的数量级相同，并且第一个值的第一位数字和第二个值的第一位数字相同，则返回 True。
    """
    loss1_first_digit = first_digit(loss1)
    loss2_first_digit = first_digit(loss2)
    same_order, order = same_order_of_magnitude(loss1, loss2)
    if loss1_first_digit == loss2_first_digit and same_order:
        if abs(loss1 - loss2) < 0.1 * order and abs(loss1 - loss2) < 1000:
            return True
    return False

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
    file_path = result_path + f"/result_{CONFIG_name}_merge.xlsx"
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.exists(file_path):
        # 创建一个DataFrame
        columns = ["Parameter", "Recall@5", "Recall@10", "Recall@15", "Recall@20",
                   "Pre@5", "Pre@10", "Pre@15", "Pre@20",
                   "NDCG@5", "NDCG@10", "NDCG@15", "NDCG@20",
                   "RMRR@5", "RMRR@10", "RMRR@15", "RMRR@20"]
        df = pd.DataFrame(columns=columns)
        df.to_excel(file_path, index=False)
    # 读取现有的Excel文件
    df = pd.read_excel(file_path)

    for lr, l2_reg, item_level_ratio, bundle_level_ratio, bundle_agg_ratio, embedding_size, num_layers, c_lambda, c_temp, LLM_name, seed in \
            product(conf['lrs'], conf['l2_regs'], conf['item_level_ratios'], conf['bundle_level_ratios'], conf['bundle_agg_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"], conf['LLM_name'], conf['seed']):
        t1 = time.time()
        run_path = f"{server_model_save_path}/runs/%s/%s" %(conf["dataset"], conf["model"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size
        conf["LLM"] = LLM_name

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
        # log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
            
        run = SummaryWriter(run_path)
        cur_best_pre_0, stopping_step = 0, 0
        print("args.pretrain\t", conf['pretrain'])
        print('no pretrain, without pretraining.')

        wandb.init(
            # set the wandb project where this run will be logged
            project=f"AC-KCCL_ctemp-{c_temp}_cweight-random_numlayer-{num_layers}"
                    f"_aug{conf['aug_type']}"
                    f"_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}"
                    f"_LLM_enhanced",
            # track hyperparameters and run metadata
            name=f"5-{LLM_name}merge-lr:{conf['lrs'][0]}_reg:{l2_reg}_ctemp-{c_temp}_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}",
            config={
                f"learning_rate": {conf['lrs'][0]},
                "architecture": "KCCL",
                "dataset": "Herb",
                "epochs": {conf['epochs']},
            }
        )
        # model
        if conf['model'] == 'KCCL':
            model = KCCL(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))

        # 检测模型参数的梯度信息
        wandb.watch(model)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        # best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        loss_loger, pre_loger, rec_loger, ndcg_loger, rmrr_loger = [], [], [], [], []

        # start a new wandb run to track this script
        paras = (f"{LLM_name}merge-lr{lr}_reg{l2_reg}_ctemp-{c_temp}_cweight-random_numlayer-{num_layers}_emb-{str(conf['embedding_sizes'])}"
                 f"_aug{conf['aug_type']}"
                 f"_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}"
                 f"_ctemp-{c_temp}_numlayer-{num_layers}_{conf['save_tail']}_{str(seed)}")
        c_lambda_weight = 1.0
        mse_loss_list = []
        # 初始化各个vector的name和热力图列表
        # vector_names = [["users_feature_0", []], ["users_feature_1", []],
        #                 ["bundles_feature_0", []], ["bundles_feature_1", []],
        #                 ["item_features_0", []], ["item_features_1", []],
        #                 ["W_predict_mlp_user_0", []], ["b_predict_mlp_user_0", []]]

        c_loss_flag = False
        # 创建一个ELU激活函数层
        # elu = nn.ELU(alpha=1.0, inplace=False)
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i
                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor+1) % ed_interval_bs == 0:
                    ED_drop = True
                item_weights = torch.tensor(dataset.item_weights, dtype=torch.float32).to(
                    conf['device'])  # [n_item, 1] 每个item在数据集中出现的频率权重
                mse_loss, c_loss, users_feature, bundles_feature, item_features = model(batch, us_u, item_weights, ED_drop=ED_drop)
                mse_loss_list.append(mse_loss.item())
                # if conf["c_lambdas"][0] == -1:
                #     c_lambda_weight = mse_loss / (c_loss - 1e-5)
                #     conf["c_lambda"] = c_lambda_weight
                if conf["c_lambdas"][0] and len(mse_loss_list) > 2 and loss_magnitude(mse_loss_list[-1], mse_loss_list[-2]):  # 趋于稳定
                    c_loss_flag = True
                if c_loss_flag:
                    awl = AutomaticWeightedLoss(2)
                    loss = awl(mse_loss, c_loss)
                else:
                    loss = mse_loss
                # loss = mse_loss
                loss.backward()
                optimizer.step()
                loss_scalar = loss.detach()
                mse_loss_scalar = mse_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_mse", mse_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f, MSE_loss: %.4f, c_loss: %.4f * %.4f" %(epoch, loss_scalar, mse_loss_scalar, c_loss_scalar, conf["c_lambda"]))

            if (epoch+1) % 10 == 0:
                ret = test(model, conf, dataset, us_u)
                rec_loger.append(ret['recall'])
                pre_loger.append(ret['precision'])
                ndcg_loger.append(ret['ndcg'])
                rmrr_loger.append(ret['rmrr'])

                perf_str = (f'Epoch {epoch} : train==[{loss.item()}={conf["c_lambda"] * mse_loss.item()} + {c_loss.item()}]\n'
                            f'recall=[{round(ret["recall"][0], 5)}, {round(ret["recall"][-1], 5)}],'
                            f'pre=[{round(ret["precision"][0], 5)}, {round(ret["precision"][-1], 5)}],'
                            f'ndcg=[{round(ret["ndcg"][0], 5)}, {round(ret["ndcg"][-1], 5)}],'
                            f'rmrr=[{round(ret["rmrr"][0], 5)}, {round(ret["rmrr"][-1], 5)}],'
                            )
                wandb.log({"loss": loss.item(),
                           "MSE_loss": mse_loss.item(),
                           "C_loss":  c_loss.item(),
                           "recall@5": ret["recall"][0],
                           # "recall@20": ret["recall"][3],
                           "pre@5": ret["precision"][0],
                           # "pre@20": ret["precision"][3],
                           "ndcg@5": ret["ndcg"][0],
                           # "ndcg@20": ret["ndcg"][3],
                           # "rmrr@5": ret["rmrr"][0],
                           # "rmrr@20": ret["rmrr"][3],

                           })
                print("paras\t", paras)
                print(perf_str)
                cur_best_pre_0, stopping_step, should_stop = no_early_stopping(ret['precision'][0], cur_best_pre_0,
                                                                                stopping_step, expected_order='acc')

                if should_stop == True:
                    print('early stopping')
                    break

                # *********************************************************
                # save the user & item embeddings for pretraining.
                if ret['precision'][0] == cur_best_pre_0 and conf['save_flag'] == 1:
                    weights_save_path = os.path.join(server_model_save_path, conf["dataset"],
                                                     conf['model'], LLM_name,
                                                     f"lr{lr}_reg{l2_reg}_aug{conf['aug_type']}"
                                                     f"_itemdr{item_level_ratio}_nodr{bundle_level_ratio}_noaggdr{bundle_agg_ratio}"
                                                     f"_ctemp-{c_temp}_numlayer-{num_layers}_{conf['save_tail']}_merge-{seed}/")
                    ensureDir(weights_save_path)
                    print("\n", "*" * 80, "model sava path", weights_save_path + 'model.pkl')
                    torch.save(model, weights_save_path + 'model.pkl')
                    print('save the weights in path: ', weights_save_path)
                    mf_best_epoch = epoch
                    print('mf loss best epoch is {} now.'.format(str(epoch+1)))

        # for vector_i in range(len(vector_names)):
        #     vector_name = vector_names[vector_i][0]
        #     # 记录每50个热力图到W&B
        #     wandb.log({f"{vector_name}]": vector_names[vector_i][1]})
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        rmrrs = np.array(rmrr_loger)

        best_pres_0 = max(pres[:, 0])
        idx = list(pres[:, 0]).index(best_pres_0)

        final_perf = "Best Iter=[%d]\trecall=[%s], precision=[%s], ndcg=[%s], rmrr= [%s]" % \
                     ((idx+1)*10, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                      '\t'.join(['%.5f' % r for r in rmrrs[idx]]))
        print(final_perf)
        # 写入文件
        new_df = pd.DataFrame([[paras,
                                      round(recs[idx][0], 5), round(recs[idx][1], 5), round(recs[idx][2], 5), round(recs[idx][3], 5),
                                      round(pres[idx][0], 5), round(pres[idx][1], 5), round(pres[idx][2], 5), round(pres[idx][3], 5),
                                      round(ndcgs[idx][0], 5), round(ndcgs[idx][1], 5), round(ndcgs[idx][2], 5), round(ndcgs[idx][3], 5),
                                      round(rmrrs[idx][0], 5), round(rmrrs[idx][1], 5), round(rmrrs[idx][2], 5), round(rmrrs[idx][3], 5)]],
                                    columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)

        # 写入Excel文件
        df.to_excel(file_path, index=False)
        print(f"New results have been added to {file_path}")
        print('end ')
        wandb.finish()
        t2 = time.time()
        print("Total time:", f"{str((t2 - t1)/60)} min")

def wandb_heat_image(x, heatmap_path_dir, epoch, vector_name, images):
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(x, cmap="viridis")

    # 保存热力图到文件
    heatmap_path = f"{heatmap_path_dir}/{vector_name}_epoch_{epoch + 1}.png"
    plt.savefig(heatmap_path)
    plt.close()

    # 将热力图添加到图像列表
    images.append(wandb.Image(heatmap_path, caption=f"Epoch {epoch + 1}"))
    return images

def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def test(model, conf, dataset, us_u):
    """
    Args:
    """
    Ks = conf["set_K"]
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'rmrr': np.zeros(len(Ks))}
    # test_users = users_to_test
    bundle_val_data = dataset.bundle_val_data
    test_us_b_pairs = np.array(bundle_val_data.us_b_pairs, dtype=np.int32)
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
                v = dataset.herbs_id_list_tune[en_i] # sym-index's true herb set list
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

def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
