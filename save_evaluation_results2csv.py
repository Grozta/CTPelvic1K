import pandas as pd
import pickle as pkl
import numpy as np
import os

if __name__ == '__main__':
    # base_dir = os.environ['HOME']
    # eval_reslult_pkl_path = base_dir + '/all_data/nnUNet/rawdata/ipcai2021_ALL_Test/' \
    #                                    'Task22_ipcai2021_T__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__fold0_3dcascadefullres_pred/' \
    #                                    'evaluation_mcr__2000_False.pkl'
    # eval_reslult_pkl_path = '/data/datasets/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_2d_pred/evaluation_sdf_35__2000.pkl'
    eval_reslult_pkl_path = '/data/datasets/CTPelvic1K/folds/fold5/test/img/Task5_CERVIX__CTPelvic1K__fold5_3dfullres_pred_hdc32/cut_spine/evaluation_sdf_35__2000.pkl'
    print(eval_reslult_pkl_path)
    with open(eval_reslult_pkl_path, 'rb') as f:
        eval_reslult = pkl.load(f)  # dict of names and quality sub-dict

    datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6']
    for dataset in datasets:
        names = []
        bone1_Hausdorff = []
        bone1_Dice = []
        bone1_Acc = []
        bone2_Hausdorff = []
        bone2_Dice = []
        bone2_Acc = []
        bone3_Hausdorff = []
        bone3_Dice = []
        bone3_Acc = []
        bone4_Hausdorff = []
        bone4_Dice = []
        bone4_Acc = []
        whole_Hausdorff = []
        whole_Dice = []
        whole_Acc = []
        mean_Hausdorff = []
        mean_Dice = []
        mean_Acc = []
        weight_Hausdorff = []
        weight_Dice = []
        weight_Acc = []
        for na, quality in eval_reslult.items():
            if not dataset in na:
                continue
            names.append(na)
            bone1_Hausdorff.append(quality[1]['Hausdorff'])
            bone1_Dice.append(quality[1]['dice'])
            bone1_Acc.append(quality[1]['acc'])
            bone2_Hausdorff.append(quality[2]['Hausdorff'])
            bone2_Dice.append(quality[2]['dice'])
            bone2_Acc.append(quality[2]['acc'])
            bone3_Hausdorff.append(quality[3]['Hausdorff'])
            bone3_Dice.append(quality[3]['dice'])
            bone3_Acc.append(quality[3]['acc'])
            bone4_Hausdorff.append(quality[4]['Hausdorff'])
            bone4_Dice.append(quality[4]['dice'])
            bone4_Acc.append(quality[4]['acc'])
            whole_Hausdorff.append(quality['whole']['Hausdorff'])
            whole_Dice.append(quality['whole']['dice'])
            whole_Acc.append(quality['whole']['acc'])
            mean_Hausdorff.append(quality['mean_hausdorff'])
            mean_Dice.append(quality['mean_dice'])
            mean_Acc.append(quality['mean_acc'])
            weight_Hausdorff.append(quality['weighted_mean_hausdorff'])
            weight_Dice.append(quality['weighted_mean_dice'])
            weight_Acc.append(quality['weighted_mean_acc'])
        if 0 == len(names):
            continue
        print(dataset, len(names))
        print(
            'bone1_Dice:', np.array(bone1_Dice).mean(), '\n',
            'bone1_Hausdorff:', np.array(bone1_Hausdorff).mean(), '\n',
            'bone1_Acc:', np.array(bone1_Acc).mean(), '\n',
            'bone2_Dice:', np.array(bone2_Dice).mean(), '\n',
            'bone2_Hausdorff:', np.array(bone2_Hausdorff).mean(), '\n',
            'bone2_Acc:', np.array(bone2_Acc).mean(), '\n',
            'bone3_Dice:', np.array(bone3_Dice).mean(), '\n',
            'bone3_Hausdorff:', np.array(bone3_Hausdorff).mean(), '\n',
            'bone3_Acc:', np.array(bone3_Acc).mean(), '\n',
            'bone4_Dice:', np.array(bone4_Dice).mean(), '\n',
            'bone4_Hausdorff:', np.array(bone4_Hausdorff).mean(), '\n',
            'bone4_Acc:', np.array(bone4_Acc).mean(), '\n',
            'whole_Dice:', np.array(whole_Dice).mean(), '\n',
            'whole_Hausdorff:', np.array(whole_Hausdorff).mean(), '\n',
            'whole_Acc:', np.array(whole_Acc).mean(), '\n',
            'mean_Dice:', np.array(mean_Dice).mean(), '\n',
            'mean_Hausdorff:', np.array(mean_Hausdorff).mean(), '\n',
            'mean_Acc:', np.array(mean_Acc).mean(), '\n',
            'weight_Dice:', np.array(weight_Dice).mean(), '\n',
            'weight_Hausdorff:', np.array(weight_Hausdorff).mean(), '\n',
            'weight_Acc:', np.array(weight_Acc).mean(), '\n',
        )
        assert len(names) == len(bone1_Dice)
        assert len(names) == len(bone1_Hausdorff)
        assert len(names) == len(bone1_Acc)
        assert len(names) == len(bone2_Dice)
        assert len(names) == len(bone2_Hausdorff)
        assert len(names) == len(bone2_Acc)
        assert len(names) == len(bone3_Dice)
        assert len(names) == len(bone3_Hausdorff)
        assert len(names) == len(bone3_Acc)
        assert len(names) == len(bone4_Dice)
        assert len(names) == len(bone4_Hausdorff)
        assert len(names) == len(bone4_Acc)
        assert len(names) == len(whole_Dice)
        assert len(names) == len(whole_Hausdorff)
        assert len(names) == len(whole_Acc)
        assert len(names) == len(mean_Dice)
        assert len(names) == len(mean_Hausdorff)
        assert len(names) == len(mean_Acc)
        assert len(names) == len(weight_Dice)
        assert len(names) == len(weight_Hausdorff)
        assert len(names) == len(weight_Acc)

        # results = {'names': names,
        #            'bone1_Dice': bone1_Dice,
        #            'bone1_Hausdorff': bone1_Hausdorff,
        #            'bone1_Acc': bone1_Acc,
        #            'bone2_Dice': bone2_Dice,
        #            'bone2_Hausdorff': bone2_Hausdorff,
        #            'bone2_Acc': bone2_Acc,
        #            'bone3_Dice': bone3_Dice,
        #            'bone3_Hausdorff': bone3_Hausdorff,
        #            'bone3_Acc': bone3_Acc,
        #            'bone4_Dice': bone4_Dice,
        #            'bone4_Hausdorff': bone4_Hausdorff,
        #            'bone4_Acc': bone4_Acc,
        #            'whole_Dice': whole_Dice,
        #            'whole_Hausdorff': whole_Hausdorff,
        #            'whole_Acc': whole_Acc,
        #            'mean_Dice': mean_Dice,
        #            'mean_Hausdorff': mean_Hausdorff,
        #            'mean_Acc': mean_Acc,
        #            'weight_Dice': weight_Dice,
        #            'weight_Hausdorff': weight_Hausdorff,
        #            'weight_Acc': weight_Acc
        #            }
        results = {'names': names,
                   'Sacrum_DC': bone1_Dice,
                   'Sacrum_HD': bone1_Hausdorff,
                   'Sacrum_Acc': bone1_Acc,
                   'Right hip_DC': bone2_Dice,
                   'Right hip_HD': bone2_Hausdorff,
                   'Right hip_Acc': bone2_Acc,
                   'Left hip_DC': bone3_Dice,
                   'Left hip_HD': bone3_Hausdorff,
                   'Left hip_Acc': bone3_Acc,
                   'Lumbar spine_DC': bone4_Dice,
                   'Lumbar spine_HD': bone4_Hausdorff,
                   'Lumbar spine_Acc': bone4_Acc,
                   'whole_DC': whole_Dice,
                   'whole_HD': whole_Hausdorff,
                   'whole_Acc': whole_Acc,
                   'mean_DC': mean_Dice,
                   'mean_HD': mean_Hausdorff,
                   'mean_Acc': mean_Acc,
                   'weight_DC': weight_Dice,
                   'weight_HD': weight_Hausdorff,
                   'weight_Acc': weight_Acc
                   }
        results_pd = pd.DataFrame(results)
        cols = list(results_pd)
        cols.remove('names')
        col_mean = results_pd[cols].mean()
        col_mean['names'] = 'average'
        results_pd = results_pd.append(col_mean, ignore_index=True)
        save_csv_path = eval_reslult_pkl_path.replace('.pkl', '_{}.csv'.format(dataset))
        results_pd.to_csv(save_csv_path)
