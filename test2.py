#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import json

if __name__ == '__main__':
    cur_dir = '/data/datasets/CTPelvic1K/all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K'
    json_file = os.path.join(cur_dir, 'dataset.json')
    with open(json_file, 'r') as f:
        result = json.load(f)
    training_list = result['training']
    test_list = result['test']
    print('test')
    for path in test_list:
        if not os.path.exists(os.path.join(cur_dir, path)):
            print(f'{path} not exists')
            break
    print('train')
    for t in training_list:
        img_path = os.path.join(cur_dir, t['image'])
        label_path = os.path.join(cur_dir, t['label'])
        if os.path.basename(img_path) != os.path.basename(label_path):
            print(f'{img_path} and {label_path} is not same')
            break
        if not os.path.exists(img_path):
            print(f'{img_path} not exists')
            break
        if not os.path.exists(label_path):
            print(f'{label_path} not exists')
            break
