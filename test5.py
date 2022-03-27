#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import pickle

if __name__ == '__main__':
    path = '/data/PyTorch_model/CTPelvic1K/3d_cascade_fullres/Task5_CERVIX/nnUNetTrainerCascadeFullRes__nnUNetPlans/fold_5/model_best.model.pkl'
    with open(path, 'rb') as f:
        temp = pickle.load(f)  # /data/PyTorch_model/CTPelvic1K
    init = list(temp['init'])
    for x in init:
        print(x)
