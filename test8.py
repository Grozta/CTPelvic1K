#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os

from utils import _sitk_Image_reader, _sitk_image_writer

if __name__ == '__main__':
    path = '/data/datasets/CTPelvic1K/folds/fold5_without4'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('_mask_4label.nii.gz'):
                cur = os.path.join(root, file)
                print(cur)
                _, pred, meta = _sitk_Image_reader(cur)
                pred[pred == 4] = 0
                _sitk_image_writer(pred, meta, cur)
