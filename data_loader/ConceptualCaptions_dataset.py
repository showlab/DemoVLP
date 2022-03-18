from base.base_dataset import TextObjectDataset
import pandas as pd
import os
import numpy as np
import random
import torch


class ConceptualCaptions3MObjectSelect(TextObjectDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """

    def __init__(self,
                 dataset_name,
                 text_params,
                 object_params,
                 data_dir,
                 object_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 mask=False):
        super(ConceptualCaptions3MObjectSelect,
              self).__init__(dataset_name, text_params, object_params,
                             data_dir, object_dir, metadata_dir, split, tsfms,
                             cut, subsample, sliding_window_stride, reader,
                             mask)
        self.object_num = self.object_params['object_num']
        self.num_frames = self.object_params['num_frames']

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data'
        split_files = {
            'train': 'cc3m_training_success_full.tsv',
            'val': 'cc3m_validation_success_full.tsv',  # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp),
                               sep='\t')

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_video_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, sample[1])
        #rel_fp = os.path.join(rel_dir, str(zlib.crc32(sample['thumbnailUrl'].encode('utf-8')) & 0xffffffff))
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]
        #return sample['caption']

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        pre = sample[1].split('_')[0]
        pre = pre.zfill(7)
        rel_object_fp = os.path.join(pre[:4], sample[1] + '_1.npz')
        full_object_fp = os.path.join(self.object_dir, self.split,
                                      rel_object_fp)

        return os.path.join(self.split, rel_object_fp), full_object_fp

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        object_rel_fp, object_fp = self._get_object_path(sample)
        caption = self._get_caption(sample)
        if not os.path.exists(object_fp):
            object_fp = object_fp.replace('_1.npz', '_0.npz')
            if not os.path.exists(object_fp):
                print("not exist object in: {}, select another sample".format(
                    object_fp))
                item_new = random.randint(1, self.__len__())
                return self.__getitem__(item_new)
        # load object
        try:
            frame_idxs = [0]
            object, object_mask, object_len = read_object_from_disk_with_object_select(
                object_fp, frame_idxs,
                self.object_num)  # [segments, topk, 2054]
        except Exception as e:
            print("Fail to load selected objects {}, object_fp {}".format(
                e, object_fp))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        meta_arr = {
            'raw_captions': caption,
            'paths': object_rel_fp,
            'dataset': self.dataset_name
        }
        data = {
            'object': object,
            'text': caption,
            'meta': meta_arr,
            'object_mask': object_mask,
            'object_len': object_len
        }
        return data


def read_object_from_disk_with_object_select(object_path, frame_idxs,
                                             object_num):
    """
    load object features and bounding box localization
    Args:
        object_path(str): absoulte path
        frame_idx: list[int]
        object_num(int): num of region
    Returns:
        feat: b x N x [2048+6]; 6 means two points and s_h, s_w
    """
    object_info = read_all_object_from_disk(object_path, frame_idxs)
    len_list = None
    object_selected = object_select_random(object_info, object_num, len_list)
    return object_selected


def object_select_random(object_info, object_num, object_num_list=None):
    """
    select loaded object
    Args:
        object_info: dict: {'frame_idx': {'feat': ndarray, 'objects_conf': ndarray, 'objects_id': ndarray, 'bbox': ndarray, 'spatial_feature': ndarray},
                            'frame_idx': ...}
        object_num: int: total number of object to load
        object_num_list: list, number of object to load for each frame
    Select rules:
        1. if object_num_list is given, choose object randomly according to the list, 
        else choose around 0.6 x object_num in each frame, if objects in current frame < 0.6 x object_num, choose all objects
    """
    o_num = int(object_num * 1.0)
    idxs = sorted(object_info.keys())
    if object_num_list is not None:
        for i, idx in enumerate(idxs):
            obj_num = object_num_list[i]
            selected_idxs = [
                random.choice(list(range(len(object_info[idx]['objects_id']))))
                for _ in range(obj_num)
            ]
            object_info[idx]['feat'] = object_info[idx]['feat'][selected_idxs]
            object_info[idx]['objects_conf'] = object_info[idx][
                'objects_conf'][selected_idxs]
            object_info[idx]['objects_id'] = object_info[idx]['objects_id'][
                selected_idxs]
            object_info[idx]['bbox'] = object_info[idx]['bbox'][selected_idxs]
            object_info[idx]['spatial_feature'] = object_info[idx][
                'spatial_feature'][selected_idxs]
            object_info[idx]['object_len'] = obj_num

            res = object_num - obj_num
            object_info[idx]['feat'] = np.pad(object_info[idx]['feat'],
                                              ((0, res), (0, 0)), 'edge')
            object_info[idx]['bbox'] = np.pad(object_info[idx]['bbox'],
                                              ((0, res), (0, 0)), 'edge')
            object_info[idx]['spatial_feature'] = np.pad(
                object_info[idx]['spatial_feature'], ((0, res), (0, 0)),
                'edge')
    else:
        for i, idx in enumerate(idxs):
            if len(object_info[idx]['objects_id']) > o_num:
                object_info[idx]['feat'] = object_info[idx]['feat'][:o_num]
                object_info[idx]['objects_conf'] = object_info[idx][
                    'objects_conf'][:o_num]
                object_info[idx]['objects_id'] = object_info[idx][
                    'objects_id'][:o_num]
                object_info[idx]['bbox'] = object_info[idx]['bbox'][:o_num]
                object_info[idx]['spatial_feature'] = object_info[idx][
                    'spatial_feature'][:o_num]
                object_info[idx]['object_len'] = o_num

                res = object_num - o_num
            else:
                res = object_num - len(object_info[idx]['objects_id'])
                object_info[idx]['object_len'] = len(
                    object_info[idx]["objects_id"])
            object_info[idx]['feat'] = np.pad(object_info[idx]['feat'],
                                              ((0, res), (0, 0)), 'edge')
            object_info[idx]['bbox'] = np.pad(object_info[idx]['bbox'],
                                              ((0, res), (0, 0)), 'edge')
            object_info[idx]['spatial_feature'] = np.pad(
                object_info[idx]['spatial_feature'], ((0, res), (0, 0)),
                'edge')

    feat_list = [object_info[i]['feat'] for i in idxs]
    spatial_feat_list = [object_info[i]['spatial_feature'] for i in idxs]
    object_len_list = [object_info[i]['object_len'] for i in idxs]
    object_mask = np.zeros((len(idxs), object_num))
    for i, length in enumerate(object_len_list):
        object_mask[i, :length] = 1
    feat = np.stack(feat_list, axis=0)
    spatial_feat = np.stack(spatial_feat_list, axis=0)
    feat_tensor = torch.from_numpy(feat)
    spatial_feat_tensor = torch.from_numpy(spatial_feat)
    object_feat = torch.cat([feat_tensor, spatial_feat_tensor], dim=-1)

    return object_feat, object_mask, object_len_list


def read_all_object_from_disk(object_path, frame_idxs):
    """
    load all object info 
    Returns:
        dict: {'frame_idx': {'feat': ndarray, 'objects_conf': ndarray, 'objects_id': ndarray, 'bbox': ndarray, 'spatial_feature': ndarray},
               'frame_idx': ...}
    """
    object_info = {}
    for index in frame_idxs:
        # print("index is: {}".format(index))
        frame_object_info = {}
        try:
            frame1 = np.load(object_path, allow_pickle=True)
            features = frame1['x']
            boxes = frame1['bbox']
            confident = frame1['info'].item()['objects_conf']
            object_ids = frame1['info'].item()['objects_id']
            confident_indices = np.argsort(confident)[::-1]
            confident = confident[confident_indices]
            boxes = boxes[confident_indices]
            features = features[confident_indices]
            object_ids = object_ids[confident_indices]

            image_width = frame1['info'].item()['image_w']
            image_height = frame1['info'].item()['image_h']
            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]
            scaled_width = box_width / image_width
            scaled_height = box_height / image_height
            scaled_x = boxes[:, 0] / image_width
            scaled_y = boxes[:, 1] / image_height
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]
            spatial_features = np.concatenate(
                (scaled_x, scaled_y, scaled_x + scaled_width,
                 scaled_y + scaled_height, scaled_width, scaled_height),
                axis=1)

            frame_object_info['feat'] = features
            frame_object_info['objects_conf'] = confident
            frame_object_info['objects_id'] = object_ids
            frame_object_info['bbox'] = boxes
            frame_object_info['spatial_feature'] = spatial_features
        except OSError:
            # if not found or npz error， return full 1 matrix
            print("object is wrong or not existed in : {}".format(object_path))
        # print(feat.size())
        object_info[index] = frame_object_info
    return object_info