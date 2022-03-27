#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import glob
import os
import re
import shutil

from batchgenerators.utilities.file_and_folder_operations import subfiles
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""
    dataset format:
        image: *_data.nii.gz
        label: *_mask_4label.nii.gz
    test: 
        image: *_data.nii.gz
    Data will be converted into a unified format used in nnunet
"""
TASK = 'Task11_CERVIX'
home_dir = '/data/datasets/CTPelvic1K'
train_dir = os.path.join(home_dir, f'all_data/nnUNet/rawdata/{TASK}')  # 训练集
test_dir = os.path.join(home_dir, f'all_data/nnUNet/rawdata/{TASK}_test')  # 测试集


def split_nii_gz(filename):
    """
    切割nii.gz的文件名与扩展名

    :param filename: nii.gz文件
    :return: (文件名,'.nii.gz')
    """
    temp, ext1 = os.path.splitext(filename)
    temp, ext2 = os.path.splitext(temp)
    return temp, ext2 + ext1


def get_filename(path):
    """
    获取nii.gz的文件名，并转换

    :param path: nii.gz文件
    :return: 转换后文件名
    """
    path_dir, filename = os.path.split(path)
    filename, ext = split_nii_gz(filename)
    # -替换为_,去掉dataset{index}_,去掉_data,
    return re.sub('_data', '', re.sub(r'^dataset\d_', '', filename.replace('-', '_')))


def get_last_dir(path):
    """
    获取最深的目录名

    :param path: 路径
    :return: 最深的目录名
    """
    return os.path.split(os.path.dirname(path))[-1]


# 先验知识： 标签的名字=dataset{index}_{训练集对应的标签的名字}_mask_4label.nii.gz
data_inform = {
    'ABDOMEN': {
        'index': 1,
        'skip': False,  # 是否跳过（不使用这个数据集）
        'img_path': '/data/datasets/Abdomen/Abdomen',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset1_mask_mappingback',  # 修改
        'patten': ['RawData/Training/img/*.nii.gz', 'RawData/Testing/img/*.nii.gz'],
        'convert_function': get_filename,  # 训练集对应的标签的名字
        'test_size': 7,
        'all_test': True  # 全部作为测试集
    },
    'COLONOG': {  # FIXME 修改convert_function,patten
        'index': 2,
        'skip': True,
        'img_path': '/data/datasets/COLONOGRAPHY',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset2_mask_mappingback',  # 修改
        'patten': [],
        'convert_function': get_filename,
        'test_size': 145,
        'all_test': True  # 全部作为测试集
    },
    'MSD_T10': {
        'index': 3,
        'skip': False,
        'img_path': '/data/datasets/MSD_T10/Task10_Colon/Task10_Colon',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset3_mask_mappingback',  # 修改
        'patten': ['imagesTr/*.nii.gz', 'imagesTs/*.nii.gz'],
        'convert_function': get_filename,
        'test_size': 31,
        'all_test': True  # 全部作为测试集
    },
    'KITS19': {
        'index': 4,
        'skip': False,
        'img_path': '/data/datasets/kits19',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset4_mask_mappingback',  # 修改
        'patten': ['*/imaging.nii.gz'],
        'convert_function': get_last_dir,
        'test_size': 9,
        'all_test': True  # 全部作为测试集
    },
    'CERVIX': {
        'index': 5,
        'skip': False,
        'img_path': '/data/datasets/CERVIX/Cervix',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset5_mask_mappingback',  # 修改
        'patten': ['RawData/Training/img/*.nii.gz', 'RawData/Testing/img/*.nii.gz'],
        'convert_function': get_filename,
        'test_size': 9,
        'all_test': False  # 全部作为测试集
    },
    'CLINIC': {
        'index': 6,
        'skip': False,
        'img_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset6_data',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset6_Anonymized_mask/ipcai2021_dataset6_Anonymized',
        # 修改
        'patten': ['*.nii.gz'],
        'convert_function': get_filename,
        'test_size': 21,
        'all_test': True  # 全部作为测试集
    },
    'CLINIC-metal': {
        'index': 7,
        'skip': False,
        'img_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset7_data',  # 修改
        'label_path': '/data/datasets/CTPelvic1K/CTPelvic1K_dataset7_mask',  # 修改
        'patten': ['*.nii.gz'],
        'convert_function': get_filename,
        'test_size': 14,
        'all_test': True  # 全部作为测试集
    }
}

if __name__ == '__main__':
    print(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    assert 0 == len(os.listdir(train_dir))
    print(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    for dataset_name, v in data_inform.items():
        print(dataset_name)
        if v['skip']:
            continue
        index = v['index']
        img_path = v['img_path']
        label_path = v['label_path']
        test_size = v['test_size']

        images_list = []  # 所有的图片（包括训练集、测试集）
        label_list = glob.glob(f'{label_path}/*_mask_4label.nii.gz')  # 标签
        for patten in v['patten']:
            images_list += glob.glob(os.path.join(img_path, patten))
        target_img_list = []  # 有标签的图片

        flag = True

        for path in images_list:
            file_name = v['convert_function'](path)
            target_label_path = f'{label_path}/dataset{index}_{file_name}_mask_4label.nii.gz'
            target_label_path2 = f'{label_path}/{file_name}_mask_4label.nii.gz'
            if not flag:
                if target_label_path2 in label_list:
                    target_img_list.append(path)
            elif target_label_path in label_list:
                target_img_list.append(path)
            else:
                if target_label_path2 in label_list:
                    flag = False
                    target_img_list.append(path)

        # 按转换后的文件名排序，保证对应
        target_img_list.sort(key=lambda x: v['convert_function'](x))
        label_list.sort(key=lambda x: os.path.basename(x))

        if v['all_test']:
            test_size = len(label_list)

        if len(label_list) > test_size:
            image_train_list, image_test_list, label_train_list, label_test_list = train_test_split(target_img_list,
                                                                                                    label_list,
                                                                                                    test_size=test_size)
        else:
            image_train_list = []
            label_train_list = []
            image_test_list = target_img_list
            label_test_list = label_list

        prefix = '' if flag else f'dataset{index}_'
        for image_filename, label_filename in tqdm(zip(image_train_list, label_train_list)):
            file_name = v['convert_function'](image_filename)
            shutil.copyfile(image_filename, os.path.join(train_dir, f'dataset{index}_{file_name}_data.nii.gz'))
            shutil.copyfile(label_filename, os.path.join(train_dir, prefix + os.path.basename(label_filename)))

        for image_filename, label_filename in tqdm(zip(image_test_list, label_test_list)):
            file_name = v['convert_function'](image_filename)
            shutil.copyfile(image_filename, os.path.join(test_dir, f'dataset{index}_{file_name}_data.nii.gz'))
            shutil.copyfile(label_filename, os.path.join(test_dir, prefix + os.path.basename(label_filename)))

    # check
    nii_files_tr_data = subfiles(train_dir, True, None, "_data.nii.gz", True)
    nii_files_tr_seg = subfiles(train_dir, True, None, "_mask_4label.nii.gz", True)
    nii_files_ts = subfiles(test_dir, True, None, "_data.nii.gz", True)
    print('checking')
    for img, seg in zip(nii_files_tr_data, nii_files_tr_seg):
        img = os.path.basename(img)
        seg = os.path.basename(seg)
        if img.replace('_data', '_mask_4label') != seg:
            print('标签和数据集对不上？？？')
            print(img, seg)
            break
