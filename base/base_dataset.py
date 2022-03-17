import random
import cv2
# import av
import os
import numpy as np
import torch
import random
from abc import abstractmethod
from torch.utils.data import Dataset


class TextObjectDataset(Dataset):

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
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.object_params = object_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        # == object dir ==
        self.object_dir = os.path.expandvars(object_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.label_type = 'caption'
        self._load_metadata()
        self.mask = mask
        self.segments = object_params['num_frames']

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError(
            "Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError(
            "Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError(
            "Get caption function must be implemented by subclass")

    def _get_object_path(self, sample, rm_split=False):
        raise NotImplementedError(
            "Get caption function must be implemented by subclass")

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0,
                                  stop=x,
                                  num=min(x, self.video_params['num_frames']) +
                                  1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.
                                sliding_window_stride))
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def _sample_objects(self,
                        num_objects,
                        vlen,
                        sample='rand',
                        fix_start=None):
        acc_samples = min(num_objects, vlen)
        intervals = np.linspace(start=0, stop=vlen,
                                num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError
        return frame_idxs

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        object_rel_fp, object_fp = self._get_object_path(sample)
        caption = self._get_caption(sample)
        if not os.path.exists(os.path.join(object_fp, '0.npz')):
            print("not exist object in: {}, select another sample".format(
                object_fp))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        # load object
        object_num = len(os.listdir(object_fp))
        if object_num < 2:
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        if self.split == 'train':
            frame_idxs = self._sample_objects(self.segments,
                                              object_num,
                                              sample='rand')
            frame_idxs = sorted(frame_idxs)
        else:
            frame_idxs = self._sample_objects(self.segments,
                                              object_num,
                                              sample='uniform')
        object = read_object_from_disk(object_fp, frame_idxs,
                                       top_k=20)  # [segments, topk, 2054]
        meta_arr = {
            'raw_captions': caption,
            'paths': object_rel_fp,
            'dataset': self.dataset_name
        }
        data = {'object': object, 'text': caption, 'meta': meta_arr}
        return data


def read_object_from_disk(object_path, frame_idxs, top_k=20, v=1):
    """
    load object features and bounding box localization
    Args:
        object_path(str): absoulte path
        top_k(int): top-k confidence regions
        v(int):  1:  select top-k confidence regions [maybe with same class] 2: select top-k confidence regions with [different classes]
    Returns:
        feat: b x N x [2048+6]; 6 means two points and s_h, s_w
    """
    feats = None
    for index in frame_idxs:
        # print("index is: {}".format(index))
        try:
            frame1 = np.load(os.path.join(object_path, '{}.npz'.format(index)),
                             allow_pickle=True)
            features = frame1['x']
            boxes = frame1['bbox']
            confident = frame1['info'].item()['objects_conf']
            condident_indices = np.argsort(confident)[::-1]
            boxes = boxes[condident_indices]
            features = features[condident_indices]
            # print(features.shape)
            object_ids = frame1['info'].item()['objects_id']
            if v == 2:
                new_object, unique_indices = np.unique(object_ids,
                                                       return_index=True)
                boxes = boxes[unique_indices]
                features = features[unique_indices]
            if boxes.shape[0] < top_k:
                res = top_k - boxes.shape[0]
                boxes = np.pad(boxes, ((0, res), (0, 0)), 'edge')
                features = np.pad(features, ((0, res), (0, 0)), 'edge')
            boxes = boxes[:top_k, :]
            features = features[:top_k, :]
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
            feat = torch.cat([
                torch.from_numpy(features),
                torch.from_numpy(spatial_features)
            ],
                             dim=1)
        except OSError:
            # if not found or npz error， return full 1 matrix
            print("object is wrong or not existed in : {}".format(
                os.path.join(object_path, '{}.npz'.format(index))))
            feat = torch.full((top_k, 2054), 1.0)
        # print(feat.size())
        if feats is None:
            feats = feat.unsqueeze(0)
        else:
            feats = torch.vstack([feats, feat.unsqueeze(0)])
    return feats
