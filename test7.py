#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from glob import glob
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

from postprocessing import newsdf_post_processor
from utils import _sitk_Image_reader, _sitk_image_writer

if __name__ == '__main__':
    path = '/data/datasets/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_3dfullres_pred'
    target = '/data/datasets/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_3dfullres_pred_out'
    os.makedirs(target, exist_ok=True)
    images = glob(os.path.join(path, '*.nii.gz'))
    images.sort()
    for file in tqdm(images):
        _, pred, meta = _sitk_Image_reader(file)
        pred_sdf = newsdf_post_processor(pred, sdf_th=0.4, region_th=2000)
        print(pred_sdf.shape, np.unique(pred_sdf))

        region_properties = regionprops(pred_sdf)
        other_min_x, other_max_x = pred.shape[0], 0
        min_x, max_x = pred.shape[0], 0
        for region in region_properties:
            x, y, z = region.coords[0]
            label = pred_sdf[x, y, z]
            if label == 4:
                min_x, _, _, max_x, _, _ = region.bbox
            else:
                x1, _, _, x2, _, _ = region.bbox
                other_min_x = min(other_min_x, x1)
                other_max_x = max(other_max_x, x2)
        pred[max(other_max_x + 12, min_x) + 1:] = 0
        _sitk_image_writer(pred, meta, os.path.join(target, os.path.basename(file)))
