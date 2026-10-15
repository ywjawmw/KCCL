#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据当前训练集生成 user_item_train.txt 和 bundle_item_train.txt。"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def build_graphs():
    records = [
        line.strip().split('\t')
        for line in (BASE_DIR / 'train_id.txt').read_text(encoding='utf-8-sig').splitlines()
        if line.strip()
    ]
    bundle_ids = [
        int(line.split()[1])
        for line in (BASE_DIR / 'set_user_bundle_train.txt').read_text(
            encoding='utf-8-sig').splitlines()
        if line.strip()
    ]
    if len(records) != len(bundle_ids):
        raise ValueError('train_id.txt and set_user_bundle_train.txt must have the same record count.')

    seen_bundle_items = set()
    with (BASE_DIR / 'user_item_train.txt').open(
            'w', encoding='utf-8', newline='\n') as user_file, (
            BASE_DIR / 'bundle_item_train.txt').open(
                'w', encoding='utf-8', newline='\n') as bundle_file:
        for (symptom_text, herb_text), bundle in zip(records, bundle_ids):
            symptoms = list(map(int, symptom_text.split()))
            herbs = list(map(int, herb_text.split()))
            # 症状与中药共现保留重复次数，治法与中药的关系每条只写一次。
            for symptom in symptoms:
                for herb in herbs:
                    user_file.write('{}\t{}\n'.format(symptom, herb))
            for herb in herbs:
                edge = (bundle, herb)
                if edge not in seen_bundle_items:
                    seen_bundle_items.add(edge)
                    bundle_file.write('{}\t{}\n'.format(bundle, herb))
    print('Generated user_item_train.txt and bundle_item_train.txt')


if __name__ == '__main__':
    build_graphs()
