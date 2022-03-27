#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from glob import glob

from tqdm import tqdm

from postprocessing import newsdf_post_processor
from utils import _sitk_Image_reader, _sitk_image_writer

if __name__ == '__main__':
    path = '/media/peng/F/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_3dfullres_pred_backup'
    target = '/media/peng/F/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_3dfullres_pred_backup_sdf'
    os.makedirs(target, exist_ok=True)
    for file in tqdm(glob(os.path.join(path, '*.nii.gz'))):
        _, pred, meta = _sitk_Image_reader(file)
        pred = newsdf_post_processor(pred, sdf_th=0.4, region_th=2000)
        _sitk_image_writer(pred, meta, os.path.join(target, os.path.basename(file)))
