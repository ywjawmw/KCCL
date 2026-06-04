#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
# from torch_geometric.nn.models import MLP
# from Attention_layer import SelfAttention
import pickle


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))    # sqrt(Neighbor)
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))     # sqrt(Neighbor)
    graph = rowsum_sqrt @ graph @ colsum_sqrt            # sqrt(Neighbor) * graph * sqrt(Neighbor)

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class KCCL(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_user_sets = conf["num_users_set"]

        assert isinstance(raw_graph, list)
        self.us_b_graph, self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.mlp_predict_weight_size = eval(str(conf['mlp_layer_size']))
        self.mlp_predict_n_layers = len(self.mlp_predict_weight_size)  # [64]

        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[
                                                 len(self.mlp_predict_weight_size) - 2]] + self.mlp_predict_weight_size \
                                            + [self.mlp_predict_weight_size[len(self.mlp_predict_weight_size) - 1]]
        # self.mlp = MLP(self.mlp_predict_weight_size)
        print('mlp_predict_weight_size_list ', self.mlp_predict_weight_size_list)  # 384 384 192 192

        # self.attention_layer = SelfAttention(self.mlp_predict_weight_size_list[0],
        #                                      attn_dropout_prob=0.0)

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

        self.init_emb()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        initializer = nn.init.xavier_uniform_
        self.weights = nn.ParameterDict()
        x = 0
        for k in range(0, self.mlp_predict_n_layers):
            W_predict_mlp_user = torch.empty(
                [self.mlp_predict_weight_size_list[x], self.mlp_predict_weight_size_list[x + 1]])
            b_predict_mlp_user = torch.empty([1, self.mlp_predict_weight_size_list[x + 1]])
            self.weights.update({'W_predict_mlp_user_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            self.weights.update({'b_predict_mlp_user_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
            x = x + 2
        if "LLM_name" in self.conf.keys():
            W_llm_adapter_user = torch.empty([self.embedding_size, self.embedding_size])
            W_llm_adapter_item = torch.empty([self.embedding_size, self.embedding_size])
            # b_llm_adapter_user = torch.empty([1, self.embedding_size])
            # b_llm_adapter_item = torch.empty([1, self.embedding_size])

            self.weights.update({'W_llm_adapter_user': nn.Parameter(initializer(W_llm_adapter_user))})
            self.weights.update({'W_llm_adapter_item': nn.Parameter(initializer(W_llm_adapter_item))})
            # self.weights.update({'b_llm_adapter_user': nn.Parameter(initializer(b_llm_adapter_user))})
            # self.weights.update({'b_llm_adapter_item': nn.Parameter(initializer(b_llm_adapter_item))})

        self.mess_dropout_k = eval(str(conf['mess_dropout_k']))


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        self.user_set_feature = nn.Parameter(torch.FloatTensor(self.num_user_sets, self.embedding_size))
        nn.init.xavier_normal_(self.user_set_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        if "LLM_name" in self.conf.keys():
            # 从pickle文件中加载LLM生成的向量
            sym_vec_file = f"./datasets/GPT/{self.conf['dataset']}/vector_{self.conf['LLM']}_symptom_merge_summary.pkl"
            self.sym_vec = torch.load(sym_vec_file).to(self.device)
            herb_vec_file = f"./datasets/GPT/{self.conf['dataset']}/vector_{self.conf['LLM']}_herb_merge_summary.pkl"
            self.herb_vec = torch.load(herb_vec_file).to(self.device)

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        us_b_graph = self.us_b_graph
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((us_b_graph.shape[0], us_b_graph.shape[0])), us_b_graph],
                                      [us_b_graph.T, sp.csr_matrix((us_b_graph.shape[1], us_b_graph.shape[1]))]])

        bi_graph = self.bi_graph
        device = self.device
        bundle_level_item_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph],
                                           [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

                item2graph = bundle_level_item_graph.tocoo()
                values = np_edge_dropout(item2graph.data, modification_ratio)
                bundle_level_item_graph = sp.coo_matrix((values, (item2graph.row, item2graph.col)), shape=item2graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)   # us-b
        self.bundle_level_item_graph = to_tensor(laplace_transform(bundle_level_item_graph)).to(device)   # b-i


    def get_bundle_level_graph_ori(self):
        us_b_graph = self.us_b_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((us_b_graph.shape[0], us_b_graph.shape[0])), us_b_graph],
                                      [us_b_graph.T, sp.csr_matrix((us_b_graph.shape[1], us_b_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)

        bi_graph = self.bi_graph
        device = self.device
        bundle_level_item_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bundle_level_item_graph_ori = to_tensor(laplace_transform(bundle_level_item_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph    # 规范化处理 1/每行加和
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def only_one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        features = torch.spmm(graph, features)
        if self.conf["aug_type"] == "MD" and not test:  # !!! important
            features = mess_dropout(features)

        features = features / 2
        all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def two_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(2):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:  # !!! important
                features = mess_dropout(features)

            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori.to(self.conf["device"]), IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):

        #  =============================  add LLM embedding  =============================
        if test:
            self.sym_vec = self.sym_vec.to(self.conf["device"])
            self.herb_vec = self.herb_vec.to(self.conf["device"])

        users_feature_KG = F.elu(
            torch.matmul(self.sym_vec, self.weights['W_llm_adapter_user']))
        items_feature_KG = F.elu(
            torch.matmul(self.herb_vec, self.weights['W_llm_adapter_item']))
        # 计算增强特征
        # 对症状特征和中药特征分别进行条件加权
        # 检查每个实体的所有特征是否全为0
        is_all_zero_user = torch.all(users_feature_KG == 0, dim=1, keepdim=True)

        # 计算增强后的症状特征
        users_feature_enhanced = torch.where(is_all_zero_user,
                                             self.users_feature,
                                             0.5 * (self.users_feature + users_feature_KG))

        # 检查每个herb的所有特征是否全为0
        is_all_zero_item = torch.all(items_feature_KG == 0, dim=1, keepdim=True)
        items_feature_enhanced = torch.where(is_all_zero_item,
                                             self.items_feature,
                                             0.5 * (self.items_feature + items_feature_KG))

        #  =============================  local SDT level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori.to(self.conf["device"]), users_feature_enhanced, items_feature_enhanced, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, users_feature_enhanced, items_feature_enhanced, self.item_level_dropout, test)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        #  ============================= global SDT level propagation =============================
        if test:
            _, BL_items_feature = self.two_propagate(self.bundle_level_item_graph_ori.to(self.conf["device"]),
                                                     self.bundles_feature,
                                                     items_feature_enhanced,
                                                     self.bundle_level_dropout,
                                                     test)
            BL_user_set_feature, BL_bundles_feature = self.only_one_propagate(
                self.bundle_level_graph_ori.to(self.conf["device"]), self.user_set_feature, self.bundles_feature,
                self.bundle_level_dropout, test)
        else:
            _, BL_items_feature = self.two_propagate(self.bundle_level_item_graph, self.bundles_feature, items_feature_enhanced, self.bundle_level_dropout, test)
            BL_user_set_feature, BL_bundles_feature = self.only_one_propagate(self.bundle_level_graph,
                                                                           self.user_set_feature,
                                                                           self.bundles_feature, self.bundle_level_dropout,
                                                                       test)
        users_feature = [IL_users_feature, BL_user_set_feature]   # 第一个是从S-H图上得到的单个症状的 embedding，第二个是从SYN-THE图上得到的证候 embedding
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]
        item_feature = [IL_items_feature, BL_items_feature]

        return users_feature, bundles_feature, item_feature


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        if pos.shape[1] == 2:
            pos = pos[:, 0, :]       # item view feature
            aug = aug[:, 0, :]       # bundle  view feature

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature, items_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature

        # [item_num, emb_size]
        IL_items_feature, BL_items_feature = items_feature

        # cl is abbr. of "contrastive loss"
        i_cross_view_cl = self.cal_c_loss(IL_items_feature, BL_items_feature)   # herb cl
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)   # syn cl
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)  # the cl

        c_losses = [u_cross_view_cl, b_cross_view_cl, i_cross_view_cl]
        # c_losses = [u_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return c_loss

    def forward(self, batch, us_u, item_weights, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users_set, bundles = batch
        # 取出bundles的第一个维度的所有值，这是正例
        positive_bundles = bundles[:, 0]
        # 由（512,1）的Tensor转为列表
        positive_bundles = positive_bundles.tolist()
        # positive_bundles中第一维的每个值是self.bi_graph中的行号, 通过行号取出值，即正例的item
        items = self.bi_graph[positive_bundles].toarray()
        # 将items转换为one-hot编码，共753个item, batch 行
        items = (items > 0).astype(int)
        # 转换为Tensor
        items = torch.tensor(items, dtype=torch.float32).to(self.device)  # [B, item_num]
        users_set = torch.squeeze(users_set)
        users = us_u[users_set]
        users_feature, bundles_feature, item_features = self.propagate()   # list 2 [(num ,64), (num, 64)]
        users_embedding = []
        ####### S-H ######## 使用MLP聚合user embedding作为user set的整体表示
        ua_embeddings = users_feature[0]
        sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
        normal_matrix = torch.reciprocal(torch.sum(users, 1))
        normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
        # 复制embedding_size列  [B, embedding_size]
        extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
        # 对应元素相乘
        user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化
        if self.conf['mlp'] == 1:  # 采用MLP聚合user embedding
            # user_embeddings = self.mlp(user_embeddings)
            # user_embeddings = F.dropout(user_embeddings, self.mess_dropout_k[0])  # 证候归纳, user set的整体表示 [B, emb]
            for k in range(0, self.mlp_predict_n_layers):
                user_embeddings = F.relu(
                    torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                    + self.weights['b_predict_mlp_user_%d' % k])
                user_embeddings = F.dropout(user_embeddings, self.mess_dropout_k[k])  # 证候归纳, user set的整体表示 [B, emb]
            users_embedding.append(user_embeddings)  # 每个元素的维度[B,emb],其中没有负的user set
        ####### Syn-P ######## 直接从Syn-P图上学习到的user set embedding
        user_set_embeddings = users_feature[1][users_set]  # [B, emb]
        users_embedding.append(user_set_embeddings)

        mse_loss = self.create_MSE_loss(items, item_weights=item_weights, user_embeddings=users_embedding,
                                        all_user_embeddins=users_feature, ia_embeddings=item_features)
        bundles_embedding = [i[bundles] for i in bundles_feature]  # 两个级别的embedding 每个元素的维度[B,emb]
        # 第一步：找到 one-hot 向量中值为 1 的索引
        item_ids = torch.argmax(items, dim=1)  # [batch_size]
        # 第二步：去除重复的用户 ID
        unique_item_ids = torch.unique(item_ids)
        items_embedding_c_loss = [i[unique_item_ids] for i in item_features]
        c_loss = self.cal_loss(users_embedding, bundles_embedding, items_embedding_c_loss)
        # c_loss = self.cal_loss(users_embedding, bundles_embedding, item_features)
        return mse_loss, c_loss, users_feature, bundles_feature, item_features

    def create_MSE_loss(self, items, item_weights, user_embeddings, all_user_embeddins,
                            ia_embeddings):
        IL_users_feature, BL_users_feature = user_embeddings
        # [bs, 1+neg_num, emb_size]
        IL_items_feature, BL_items_feature = ia_embeddings

        IL_predict_probs = torch.matmul(IL_users_feature, IL_items_feature.transpose(0, 1))
        BL_predict_probs = torch.matmul(BL_users_feature, BL_items_feature.transpose(0, 1))


        predict_probs = torch.sigmoid(IL_predict_probs + BL_predict_probs)
        mf_loss = torch.sum(torch.matmul(torch.square((items - predict_probs)), item_weights), 0)
        # mf_loss = nn.MSELoss(reduction='elementwise_mean')(items, predict_probs)
        mf_loss = mf_loss / items.shape[0]
        regularizer_init = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(self.conf['device'])
        for i in range(0, len(all_user_embeddins)):
            all_item_embeddins = ia_embeddings[i]
            regularizer = torch.norm(all_user_embeddins[i]) ** 2 / 2 + torch.norm(all_item_embeddins[i]) ** 2 / 2
            regularizer = regularizer.reshape(1)
            # F.normalize(all_user_embeddins, p=2) + F.normalize(all_item_embeddins, p=2)
            regularizer = regularizer / items.shape[0]
            regularizer_init += regularizer

        emb_loss = self.conf['l2_regs'][0] * regularizer_init / len(all_user_embeddins)

        reg_loss = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(self.conf['device']) # 0.0
        rec_loss = mf_loss + emb_loss + reg_loss
        return rec_loss

    def evaluate(self, propagate_result, users_set, us_u, bundles):
        users_feature, bundles_feature, _ = propagate_result
        k = 0
        users_embedding = []
        users_set = torch.squeeze(users_set)
        users = us_u[users_set]
        for ua_embeddings in users_feature:
            sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
            normal_matrix = torch.reciprocal(torch.sum(users, 1))
            normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
            # 复制embedding_size列  [B, embedding_size]
            extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
            # 对应元素相乘
            user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化
            if self.conf['mlp'] == 1:  # 采用MLP聚合user embedding
                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                        + self.weights['b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout_k[k])  # 证候归纳, user set的整体表示 [B, emb]
                # for k in range(0, self.mlp_predict_n_layers):
            # list 2[(B, 2, 384), (B, 2, 384)]: 表明是两个视角的user embedding， 拿出来B个，复制n次每一个list都是： [u, f] --> [batch, f]  --> [batch, n, f] 每一行复制了n(2)遍
            # user_embeddings = torch.unsqueeze(user_embeddings, 1)
            users_embedding.append(user_embeddings)
            k += 1
        users_feature_atom, users_feature_non_atom = users_embedding  # batch_f
        # bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # b_f
        bundles_feature_atom, bundles_feature_non_atom = [i[bundles] for i in bundles_feature] # b_f
        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) \
                 + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())  # batch_b

        return scores


    def evaluate_items(self, propagate_result, users_set, us_u):
        print("*" * 100, "考虑证候-治法图！")
        users_feature, _, items_feature = propagate_result
        # k = 0
        users_embedding = []
        users_set = torch.squeeze(users_set)
        users = us_u[users_set]
        ####### S-H ######## 使用MLP聚合user embedding作为user set的整体表示
        ua_embeddings = users_feature[0]
        sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
        normal_matrix = torch.reciprocal(torch.sum(users, 1))
        normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
        # 复制embedding_size列  [B, embedding_size]
        extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
        # 对应元素相乘
        user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化
        if self.conf['mlp'] == 1:  # 采用MLP聚合user embedding
            # user_embeddings = self.mlp(user_embeddings)
            # user_embeddings = F.dropout(user_embeddings, self.mess_dropout_k[0])  # 证候归纳, user set的整体表示 [B, emb]
            for k in range(0, self.mlp_predict_n_layers):
                user_embeddings = F.relu(
                    torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                    + self.weights['b_predict_mlp_user_%d' % k])
                user_embeddings = F.dropout(user_embeddings, self.mess_dropout_k[k])  # 证候归纳, user set的整体表示 [B, emb]
            users_embedding.append(user_embeddings)  # 每个元素的维度[B,emb],其中没有负的user set
        ####### Syn-P ######## 直接从Syn-P图上学习到的user set embedding
        user_set_embeddings = users_feature[1][users_set]  # [B, emb]
        users_embedding.append(user_set_embeddings)
        users_feature_atom, users_feature_non_atom = users_embedding  # batch_f
        # bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # b_f
        items_feature_atom, items_feature_non_atom = items_feature # i_f
        scores = torch.mm(users_feature_atom, items_feature_atom.t()) \
                 + torch.mm(users_feature_non_atom, items_feature_non_atom.t())  # batch_b
        scores = torch.sigmoid(scores)
        return scores
