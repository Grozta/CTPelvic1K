#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os

from batchgenerators.utilities.file_and_folder_operations import subfiles

if __name__ == '__main__':
    home_dir = '/data/datasets/CTPelvic1K'
    train_dir = os.path.join(home_dir, 'all_data/nnUNet/rawdata/Task11_CTPelvic1K')
    output_dir = os.path.join(home_dir, 'all_data/nnUNet/nnUNet_raw/Task11_CTPelvic1K')
    test_dir = "/data/datasets/CTPelvic1K/all_data/nnUNet/rawdata/Task11_CTPelvic1K_test"

    nii_files_tr_data = subfiles(train_dir, True, None, "_data.nii.gz", True)
    nii_files_tr_seg = subfiles(train_dir, True, None, "_mask_4label.nii.gz", True)

    nii_files_ts = subfiles(test_dir, True, None, "_data.nii.gz", True)
    for img, seg in zip(nii_files_tr_data, nii_files_tr_seg):
        img = os.path.basename(img)
        seg = os.path.basename(seg)
        if img.replace('_data', '_mask_4label') != seg:
            print(img, seg)
            break

    print(len(nii_files_tr_data))
    print(len(nii_files_tr_seg))
    print(len(nii_files_ts))
