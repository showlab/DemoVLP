from base import MultiDistBaseDataLoaderExplicitSplit
from data_loader.transforms import init_transform_dict
from data_loader.MSRVTT_dataset import MSRVTTObjectSelect, MSRVTTQAObjectSelect, MSRVTTMCObjectSelect
from data_loader.WebVid_dataset import WebVidObjectSelect
from data_loader.ConceptualCaptions_dataset import ConceptualCaptions3MObjectSelect
from data_loader.DiDeMo_dataset import DiDeMoObjectSelect
from data_loader.MSVD_dataset import MSVDObjectSelect, MSVDQAObjectSelect
from data_loader.LSMDC_dataset import LSMDCObjectSelect, LSMDCMCObjectSelect
from data_loader.TGIF_dataset import TGIFFrameObjectSelect


def dataset_object_loader(dataset_name,
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
                          reader='cv2'):
    kwargs = dict(dataset_name=dataset_name,
                  text_params=text_params,
                  object_params=object_params,
                  data_dir=data_dir,
                  object_dir=object_dir,
                  metadata_dir=metadata_dir,
                  split=split,
                  tsfms=tsfms,
                  cut=cut,
                  subsample=subsample,
                  sliding_window_stride=sliding_window_stride,
                  reader=reader)

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "WebVidObjectSelect":
        dataset = WebVidObjectSelect(**kwargs)
    elif dataset_name == 'MSRVTTObjectSelect':
        dataset = MSRVTTObjectSelect(**kwargs)
    elif dataset_name == 'MSRVTTQAObjectSelect':
        dataset = MSRVTTQAObjectSelect(**kwargs)
    elif dataset_name == 'MSRVTTMCObjectSelect':
        dataset = MSRVTTMCObjectSelect(**kwargs)
    elif dataset_name == 'ConceptualCaptions3MObjectSelect':
        dataset = ConceptualCaptions3MObjectSelect(**kwargs)
    elif dataset_name == 'MSVDObjectSelect':
        dataset = MSVDObjectSelect(**kwargs)
    elif dataset_name == 'MSVDQAObjectSelect':
        dataset = MSVDQAObjectSelect(**kwargs)
    elif dataset_name == 'DiDeMoObjectSelect':
        dataset = DiDeMoObjectSelect(**kwargs)
    elif dataset_name == 'LSMDCObjectSelect':
        dataset = LSMDCObjectSelect(**kwargs)
    elif dataset_name == 'LSMDCMCObjectSelect':
        dataset = LSMDCMCObjectSelect(**kwargs)
    elif dataset_name == 'TGIFFrameObjectSelect':
        dataset = TGIFFrameObjectSelect(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class MultiDistTextObjectVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):

    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 object_params,
                 data_dir,
                 object_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_object_loader(dataset_name, text_params,
                                        object_params, data_dir, object_dir,
                                        metadata_dir, split, tsfm, cut,
                                        subsample, sliding_window_stride,
                                        reader)
        if split != 'train':
            shuffle = False
        # print(batch_size)
        # print(num_workers)
        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name
