#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from glob import glob
import os
import shutil

if __name__ == '__main__':
    path = '/data/datasets/CTPelvic1K/all_data/nnUNet/nnUNet_processed/Task5_CERVIX/nnUNet_stage1'

    for file in glob(os.path.join(path, '*_backup.pkl')):
        origin = file.replace('_backup.pkl', '.pkl')
        os.remove(origin)
        shutil.move(file, origin)
