# -*- coding: utf-8 -*-
# @Time    : 2025/11/3 10:21
# @Author  : Ywj
# @File    : config_1.py
# @Description :

CONFIG = {
    # 最佳参数
    "CONFIG_reg:7e-3-pair-MD-best":
        {
            "data_path": './datasets',
            "batch_size_train": 1024,  # the batch size for training
            "batch_size_test": 1024,  # the batch size for testing
            "topk": [10, 20, 40, 80],  # the topks metrics for evaluation
            "neg_num": 1,  # number of negatives used for BPR loss. All the experiments use 1.

            # search hyperparameters
            # the following are the best settings
            "aug_type": "MD",  # options: ED, MD, OP
            "ed_interval": 1,  # by how many epochs to dropout edge, default is 1
            "embedding_sizes": [64],  # the embedding size for user, bundle, and item
            "num_layerss": [1],
            # number of layers for the infomation progagation over the item- and bundle-level graphs

            # the following dropout rates are with respect to the "aug_type", i.e., if aug_type  is ED, the following dropout rates are for ED.
            "item_level_ratios": [0.2],  # the dropout ratio for item-view graph
            "bundle_level_ratios": [0.15, 0.2],  # the dropout ratio for bundle-view graph
            "bundle_agg_ratios": [0.2, 0.1],  # the dropout ratio for bundle-item affiliation graph

            "lrs": [2e-4, 1e-4, 3e-4],  # learning rate 3.0e-4, 5e-4, 2e-5
            "l2_regs": [7.0e-3],  # the l2 regularization weight: lambda_2
            "c_lambdas": [-1],  # the contrastive loss weight: lambda_1
            "c_temps": [0.15],  # the temperature in the contrastive loss: tau

            "epochs": 2000,  # number of epochs to train
            "test_interval": 5,  # by how many epochs to run the validation and testing.

            "mlp": 1,
            'mlp_layer_size': [64],
            'mess_dropout_k': [0.0, 0.0],
            'set_K': [5, 10, 15, 20],
            'pretrain': 0,
            'save_flag': 1,
            'seed': [3407],
            'LLM_name': ["gpt-3.5-turbo"],
            'save_tail': 'demo-MD-best'
        },

}

