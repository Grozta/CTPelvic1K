from copy import deepcopy

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from skimage.morphology import label, ball
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening
import numpy as np


# from batchgenerators.transforms import AbstractTransform


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(AbstractTransform):
    def __init__(self, channel_idx, key="data", p_per_sample=0.2, fill_with_other_class_p=0.25,
                 dont_do_if_covers_more_than_X_percent=0.25):
        """
        :param dont_do_if_covers_more_than_X_percent: dont_do_if_covers_more_than_X_percent=0.25 is 25\%!
        :param channel_idx: can be list or int
        :param key:
        """
        self.dont_do_if_covers_more_than_X_percent = dont_do_if_covers_more_than_X_percent
        self.fill_with_other_class_p = fill_with_other_class_p
        self.p_per_sample = p_per_sample
        self.key = key
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in self.channel_idx:
                    workon = np.copy(data[b, c])
                    num_voxels = np.prod(workon.shape)
                    lab, num_comp = label(workon, return_num=True)
                    if num_comp > 0:
                        component_ids = []
                        component_sizes = []
                        for i in range(1, num_comp + 1):
                            component_ids.append(i)
                            component_sizes.append(np.sum(lab == i))
                        component_ids = [i for i, j in zip(component_ids, component_sizes) if
                                         j < num_voxels * self.dont_do_if_covers_more_than_X_percent]
                        # _ = component_ids.pop(np.argmax(component_sizes))
                        # else:
                        #    component_ids = list(range(1, num_comp + 1))
                        if len(component_ids) > 0:
                            random_component = np.random.choice(component_ids)
                            data[b, c][lab == random_component] = 0
                            if np.random.uniform() < self.fill_with_other_class_p:
                                other_ch = [i for i in self.channel_idx if i != c]
                                if len(other_ch) > 0:
                                    other_class = np.random.choice(other_ch)
                                    data[b, other_class][lab == random_component] = 1
        data_dict[self.key] = data
        return data_dict


class MoveSegAsOneHotToData(AbstractTransform):
    def __init__(self, channel_id, all_seg_labels, key_origin="seg", key_target="data", remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.all_seg_labels = all_seg_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_id = channel_id

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        seg = origin[:, self.channel_id:self.channel_id + 1]
        seg_onehot = np.zeros((seg.shape[0], len(self.all_seg_labels), *seg.shape[2:]), dtype=seg.dtype)
        for i, l in enumerate(self.all_seg_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1
        target = np.concatenate((target, seg_onehot), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin

        return data_dict


class ApplyRandomBinaryOperatorTransform(AbstractTransform):
    def __init__(self, channel_idx, p_per_sample=0.3,
                 any_of_these=(binary_dilation, binary_erosion, binary_closing, binary_opening),
                 key="data", strel_size=(1, 10)):
        """

        :param channel_idx: can be list or int
        :param p_per_sample:
        :param any_of_these:
        :param fill_diff_with_other_class:
        :param key:
        :param strel_size:
        """
        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

        assert not isinstance(channel_idx, tuple), "bäh"

        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                ch = deepcopy(self.channel_idx)
                np.random.shuffle(ch)
                for c in ch:
                    operation = np.random.choice(self.any_of_these)
                    selem = ball(np.random.uniform(*self.strel_size))
                    workon = np.copy(data[b, c]).astype(int)
                    res = operation(workon, selem).astype(workon.dtype)
                    data[b, c] = res

                    # if class was added, we need to remove it in ALL other channels to keep one hot encoding
                    # properties
                    # we modify data
                    other_ch = [i for i in ch if i != c]
                    if len(other_ch) > 0:
                        was_added_mask = (res - workon) > 0
                        for oc in other_ch:
                            data[b, oc][was_added_mask] = 0
                        # if class was removed, leave it at backgound
        data_dict[self.key] = data
        return data_dict


class MoveLastFewDataToSeg_pbl(AbstractTransform):
    """
    used when there introduce other channel data. like contour, sdf...
    """

    def __init__(self, channel_ids, key_origin="data", key_target="seg", remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_ids = channel_ids

        assert isinstance(self.channel_ids, list), self.channel_ids

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)

        if np.all(np.array(self.channel_ids) < 0):
            self.channel_ids = [i + origin.shape[1] for i in self.channel_ids]
        assert not (np.any(np.array(self.channel_ids) >= 0) and np.any(np.array(self.channel_ids) < 0))

        data_o = origin[:, self.channel_ids]
        target = np.concatenate((target, data_o), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [i for i in range(origin.shape[1]) if not i in self.channel_ids]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin

        return data_dict
