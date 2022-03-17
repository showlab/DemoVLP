from base.base_dataset import TextObjectDataset
import pandas as pd
import os
import json
import numpy as np
import random
import torch
from utils.util import load_jsonl, load_json


class MSRVTTObjectSelect(TextObjectDataset):

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
        super(MSRVTTObjectSelect,
              self).__init__(dataset_name, text_params, object_params,
                             data_dir, object_dir, metadata_dir, split, tsfms,
                             cut, subsample, sliding_window_stride, reader,
                             mask)
        self.object_num = self.object_params['object_num']

    def _load_metadata(self):
        json_fp = os.path.join(self.metadata_dir, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(self.metadata_dir, 'high-quality',
                                 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        if self.cut == "miech":
            train_list_path = "train_list_miech.txt"
            test_list_path = "test_list_miech.txt"
        elif self.cut == "jsfusion":
            train_list_path = "train_list_jsfusion.txt"
            test_list_path = "val_list_jsfusion.txt"
            js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
        elif self.cut in {"full-val", "full-test"}:
            train_list_path = "train_list_full.txt"
            if self.cut == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        elif self.cut in challenge_splits:
            train_list_path = "train_list.txt"
            if self.cut == "val":
                test_list_path = f"{self.cut}_list.txt"
            else:
                test_list_path = f"{self.cut}.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(self.cut))

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path),
                               names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path),
                              names=['videoid'])
        self.split_sizes = {
            'train': len(train_df),
            'val': len(test_df),
            'test': len(test_df)
        }

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(
                np.load(os.path.join(split_dir, js_test_cap_idx_path),
                        allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(
                lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']

        self.metadata = pd.DataFrame({'captions': self.metadata})
        print("load split {}, {} samples".format(self.split,
                                                 len(self.metadata)))

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all',
                            sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        real_path = sample.name
        full_object_fp = os.path.join(self.object_dir, real_path)
        return real_path, full_object_fp

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
        object_file_num = len(os.listdir(object_fp))
        if object_file_num < 2:
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        try:
            if self.split == 'train':
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='rand')
                frame_idxs = sorted(frame_idxs)
            else:
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='uniform')
            object, object_mask, object_len = read_object_from_disk_with_object_select(
                object_fp, frame_idxs,
                self.object_num)  # [segments, topk, 2054]
        except Exception as e:
            print(
                "Fail to load selected objects {}, object_fp {}, frame_idx {}".
                format(e, object_fp, frame_idxs))
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


class MSRVTTQAObjectSelect(TextObjectDataset):

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
        super(MSRVTTQAObjectSelect,
              self).__init__(dataset_name, text_params, object_params,
                             data_dir, object_dir, metadata_dir, split, tsfms,
                             cut, subsample, sliding_window_stride, reader,
                             mask)
        self.object_num = self.object_params['object_num']

    def _load_metadata(self):
        self.metadata_dir = './meta_data'
        ans2label_file = os.path.join(self.metadata_dir,
                                      "msrvtt_train_ans2label.json")
        self.ans2label = load_json(ans2label_file)
        split_files = {
            'train': "msrvtt_qa_train.jsonl",
            'test': "msrvtt_qa_test.jsonl",
            'val': "msrvtt_qa_val.jsonl"
        }
        target_split_fp = split_files[self.split]
        meta_file = os.path.join(self.metadata_dir, target_split_fp)
        raw_datalist = load_jsonl(meta_file)
        data_size = len(raw_datalist)
        if self.subsample < 1:
            data_size = int(data_size * self.subsample)

        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:data_size]

        datalist = []
        qid = 0
        for raw_d in raw_datalist:
            d = dict(
                question=raw_d["question"],
                vid_id=raw_d["video_id"],
                answer=raw_d["answer"],  # int or str
                question_id=qid,  # be careful, it is not unique across splits
                answer_type=raw_d["answer_type"])
            qid += 1
            datalist.append(d)

        self.metadata = datalist
        self.num_labels = len(self.ans2label)
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.qid2data = {d["question_id"]: d for d in self.metadata}

        print("load split {}, {} samples".format(self.split,
                                                 len(self.metadata)))

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample['vid_id'] +
                            '.mp4'), sample['vid_id'] + '.mp4'

    def _get_question(self, sample):
        return sample['question']

    def _get_label(self, sample):
        if self.split == 'train':
            return self.ans2label[sample['answer']]
        else:
            return -1

    def _get_question_id(self, sample):
        return sample['question_id']

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        real_path = sample['vid_id']
        full_object_fp = os.path.join(self.object_dir, real_path)
        return real_path, full_object_fp

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        object_rel_fp, object_fp = self._get_object_path(sample)
        question = self._get_question(sample)
        label = self._get_label(sample)
        question_id = self._get_question_id(sample)
        if not os.path.exists(os.path.join(object_fp, '0.npz')):
            print("not exist object in: {}, select another sample".format(
                object_fp))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        # load object
        object_file_num = len(os.listdir(object_fp))
        if object_file_num < 2:
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        try:
            if self.split == 'train':
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='rand')
                frame_idxs = sorted(frame_idxs)
            else:
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='uniform')
            object, object_mask, object_len = read_object_from_disk_with_object_select(
                object_fp, frame_idxs,
                self.object_num)  # [segments, topk, 2054]
        except Exception as e:
            print("Fail to load selected objects {}, object_fp {}".format(
                e, object_fp))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        meta_arr = {
            'raw_questions': question,
            'paths': object_rel_fp,
            'dataset': self.dataset_name
        }
        data = {
            'object': object,
            'text': question,
            'meta': meta_arr,
            'object_mask': object_mask,
            'object_len': object_len,
            'label': label,
            'question_id': question_id
        }
        return data


class MSRVTTMCObjectSelect(TextObjectDataset):

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
        super(MSRVTTMCObjectSelect,
              self).__init__(dataset_name, text_params, object_params,
                             data_dir, object_dir, metadata_dir, split, tsfms,
                             cut, subsample, sliding_window_stride, reader,
                             mask)
        self.object_num = self.object_params['object_num']

    def _load_metadata(self):
        self.metadata_dir = './meta_data'
        meta_file = os.path.join(self.metadata_dir, "msrvtt_mc_test.jsonl")
        raw_datalist = load_jsonl(meta_file)
        data_size = len(raw_datalist)
        if self.subsample < 1:
            data_size = int(data_size * self.subsample)

        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:data_size]
        print(f"Loaded data size {data_size}")
        datalist = []
        for raw_d in raw_datalist:
            d = dict(
                id=raw_d["qid"],
                vid_id=raw_d["clip_name"],
                answer=raw_d["answer"],
                options=raw_d["options"],
            )
            datalist.append(d)
        self.metadata = datalist
        self.id2answer = {d["id"]: int(d["answer"]) for d in self.metadata}
        self.id2data = {d["id"]: d for d in self.metadata}

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample['vid_id'] +
                            '.mp4'), sample['vid_id'] + '.mp4'

    def _get_options(self, sample):
        return sample['options']

    def _get_label(self, sample):
        return sample['answer']

    def _get_mc_id(self, sample):
        return sample['id']

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        real_path = sample['vid_id']
        full_object_fp = os.path.join(self.object_dir, real_path)
        return real_path, full_object_fp

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata[item]
        object_rel_fp, object_fp = self._get_object_path(sample)
        label = self._get_label(sample)
        mc_id = self._get_mc_id(sample)
        options = self._get_options(sample)
        if not os.path.exists(os.path.join(object_fp, '0.npz')):
            print("not exist object in: {}, select another sample".format(
                object_fp))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        # load object
        object_file_num = len(os.listdir(object_fp))
        if object_file_num < 2:
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        try:
            if self.split == 'train':
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='rand')
                frame_idxs = sorted(frame_idxs)
            else:
                frame_idxs = self._sample_objects(self.segments,
                                                  object_file_num,
                                                  sample='uniform')
            object, object_mask, object_len = read_object_from_disk_with_object_select(
                object_fp, frame_idxs,
                self.object_num)  # [segments, topk, 2054]
        except Exception as e:
            print(
                "Fail to load selected objects {}, object_fp {}, frame_idx {}".
                format(e, object_fp, frame_idxs))
            item_new = random.randint(1, self.__len__())
            return self.__getitem__(item_new)
        meta_arr = {'paths': object_rel_fp, 'dataset': self.dataset_name}
        data = {
            'object': object,
            'text': options,
            'meta': meta_arr,
            'object_mask': object_mask,
            'object_len': object_len,
            'label': label,
            'mc_id': mc_id
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
            frame1 = np.load(os.path.join(object_path, '{}.npz'.format(index)),
                             allow_pickle=True)
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
            print("object is wrong or not existed in : {}".format(
                os.path.join(object_path, '{}.npz'.format(index))))
        # print(feat.size())
        object_info[index] = frame_object_info
    return object_info