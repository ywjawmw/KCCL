# -*- coding: utf-8 -*-
"""Testing configurations derived from the training configurations."""

from copy import deepcopy

from config_weight import CONFIG as TRAIN_CONFIG


CONFIG = deepcopy(TRAIN_CONFIG)

for config in CONFIG.values():
    config["pretrain"] = 1
    config["save_flag"] = 0
