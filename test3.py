#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import pickle
import shutil


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def check():
    pre_data_path = '/data/datasets/CTPelvic1K/pre_data'
    files = os.listdir(pre_data_path)
    files.sort()
    splits_final = load_pickle('splits_final.pkl')
    for i in range(1, 7):
        if 2 == i:
            continue
        for prefix in ['train', 'val', 'test']:
            for path in splits_final[i][prefix]:
                path = path.replace('train_', '', 1)
                flag = False
                for file in files:
                    if file.startswith(path):
                        flag = True
                        break
                if not flag:
                    print(path)


if __name__ == '__main__':
    fold_path = '/data/datasets/CTPelvic1K/folds'
    splits_final = load_pickle('splits_final.pkl')
    pre_data_path = '/data/datasets/CTPelvic1K/pre_data'
    files = os.listdir(pre_data_path)
    files.sort()
    for i in range(1, 7):
        if 2 == i:
            continue
        fold = os.path.join(fold_path, 'fold{}'.format(i))
        os.makedirs(fold, exist_ok=True)
        for prefix in ['train', 'val', 'test']:
            cur_path = os.path.join(fold, prefix)
            os.makedirs(cur_path, exist_ok=True)
            for path in splits_final[i][prefix]:
                path = path.replace('train_', '', 1)
                targets = []
                for file in files:
                    if file.startswith(path):
                        targets.append(file)
                        if len(targets) == 2:
                            break
                for target in targets:
                    shutil.copyfile(os.path.join(pre_data_path, target), os.path.join(cur_path, target))
