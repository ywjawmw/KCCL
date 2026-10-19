#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机划分 LungCancer 并生成运行文件；"""

import pickle
import random
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SPLIT_RATIO = (80, 10, 10)


def read_records():
    records = []
    for line in (BASE_DIR / 'demo.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            symptom_text, herb_text = line.strip().split('\t')
            records.append((
                tuple(map(int, symptom_text.split())),
                tuple(map(int, herb_text.split())),
            ))
    return records


def split_records(records):
    shuffled = records.copy()
    random.shuffle(shuffled)
    missing_symptoms = {s for symptoms, _ in records for s in symptoms}
    missing_herbs = {h for _, herbs in records for h in herbs}
    required, remaining = [], []
    # 先将覆盖全部症状和中药所需的样本放入训练集。
    for symptoms, herbs in shuffled:
        if missing_symptoms.intersection(symptoms) or missing_herbs.intersection(herbs):
            required.append((symptoms, herbs))
            missing_symptoms.difference_update(symptoms)
            missing_herbs.difference_update(herbs)
        else:
            remaining.append((symptoms, herbs))

    train_size = round(len(records) * SPLIT_RATIO[0] / sum(SPLIT_RATIO))
    tune_size = round(len(records) * SPLIT_RATIO[1] / sum(SPLIT_RATIO))
    fill_size = train_size - len(required)
    train = required + remaining[:fill_size]
    random.shuffle(train)
    return {
        'train': train,
        'tune': remaining[fill_size:fill_size + tune_size],
        'test': remaining[fill_size + tune_size:],
    }


def write_lines(name, lines):
    with (BASE_DIR / name).open('w', encoding='utf-8', newline='\n') as handle:
        for line in lines:
            handle.write(line + '\n')


def main():
    records = read_records()
    splits = split_records(records)
    symptom_sets, bundles = {}, {}
    for symptoms, herbs in records:
        symptom_sets.setdefault(tuple(sorted(symptoms)), len(symptom_sets))
        bundles.setdefault(tuple(sorted(herbs)), len(bundles))
    num_symptoms = max(s for symptoms, _ in records for s in symptoms) + 1
    num_herbs = max(h for _, herbs in records for h in herbs) + 1

    def record_line(record):
        symptoms, herbs = record
        return ' '.join(map(str, symptoms)) + '\t' + ' '.join(map(str, herbs))

    write_lines('LungCancer_data_size.txt', [
        '{} {} {} {}'.format(num_symptoms, len(bundles), num_herbs, len(symptom_sets))
    ])
    for split, cases in splits.items():
        write_lines(split + '_id.txt', map(record_line, cases))
        write_lines('set_user_bundle_' + split + '.txt', (
            '{}\t{}'.format(symptom_sets[tuple(sorted(symptoms))], bundles[tuple(sorted(herbs))])
            for symptoms, herbs in cases
        ))
        write_lines('user_bundle_' + split + '.txt', (
            '{}\t{}'.format(s, bundles[tuple(sorted(herbs))])
            for symptoms, herbs in cases for s in symptoms
        ))
    write_lines('valid_id.txt', map(record_line, splits['tune']))

    us_u = np.zeros((len(symptom_sets), num_symptoms), dtype=np.float32)
    for symptoms, index in symptom_sets.items():
        us_u[index, list(symptoms)] = 1
    with (BASE_DIR / 'us_u_list').open('wb') as handle:
        pickle.dump(us_u, handle, protocol=4)

    print('train={}, valid={}, test={}'.format(
        len(splits['train']), len(splits['tune']), len(splits['test'])
    ))


if __name__ == '__main__':
    main()
