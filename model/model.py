import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel
import torch
import timm
from model.object_transformer import ObjectTransformer, load_clip_pt_weight
from model.video_qa_mdoel import BUTDQAHead


class ObjectRelation(BaseModel):

    def __init__(self,
                 object_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()
        self.text_params = text_params
        self.object_params = object_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError(
                "Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        self.object_model = load_clip_pt_weight(
            ObjectTransformer(input_dim=2054,
                              region_nums=self.object_params['object_num'],
                              output_dim=256,
                              time_module=self.object_params['time_module'],
                              num_frames=self.object_params['num_frames']))

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim),
            )
        else:
            print(projection)
            raise NotImplementedError
        self.txt_proj = txt_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint,
                                    map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(
                state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            try:
                self.load_state_dict(new_state_dict, strict=True)
                print("Loaded state_dict...")
            except Exception as e:
                print("Parameters of model and state_dict are mismatched. {}".
                      format(e))
                self.load_state_dict_with_mismatch(new_state_dict)

        self.segments = self.object_params['num_frames']
        self.projection_dim = 256

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        object_data = data['object']
        object_mask = data['object_mask']
        global_text_embeddings, local_text_embeddings = self.compute_text(
            text_data)
        global_object_embeddings, local_object_embeddings, object_mask = self.compute_object(
            object_data, object_mask)
        return dict(
            global_text_embeddings=global_text_embeddings.contiguous(),
            local_text_embeddings=local_text_embeddings.contiguous(),
            global_object_embeddings=global_object_embeddings.contiguous(),
            local_object_embeddings=local_object_embeddings.contiguous(),
            object_mask=object_mask[:, 1:, ...].contiguous(),
        )

    def compute_text(self, text_data, pad=False):
        text_embeddings_all = self.text_model(**text_data).last_hidden_state
        text_embeddings = self.txt_proj(text_embeddings_all)
        global_text_embeddings, local_text_embeddings = text_embeddings[:,0,...], text_embeddings[:,1:,...]
        return global_text_embeddings, local_text_embeddings

    def compute_object(self, object_data, object_mask):
        object_embeddings, object_mask = self.object_model(
            object_data, object_mask)
        global_object_embeddings, local_object_embeddings = object_embeddings[:,0,...], object_embeddings[:,1:,...]
        return global_object_embeddings, local_object_embeddings, object_mask

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'object_model.temporal_embed' in new_state_dict and 'object_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['object_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.object_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(
                        f'### loaded {self.object_params["model"]} model has MORE frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    new_temporal_embed = load_temporal_embed[:, :
                                                             curr_num_frames, :]
                else:
                    print(
                        f'### loaded {self.object_params["model"]} model has FEWER frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([
                            load_temporal_embed.shape[0], curr_num_frames,
                            embed_dim
                        ])
                        new_temporal_embed[:, :
                                           load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed, (curr_num_frames, embed_dim),
                            mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict[
                    'object_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'object_model.custom_pos_embed' in new_state_dict and 'object_model.custom_pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['object_model.custom_pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['object_model.custom_pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.'
                )
        return new_state_dict

    def load_state_dict_with_mismatch(self, loaded_state_dict_or_path):
        """operated in-place, no need to return `model`"""

        if isinstance(loaded_state_dict_or_path, str):
            loaded_state_dict = torch.load(loaded_state_dict_or_path,
                                           map_location="cpu")
        else:
            loaded_state_dict = loaded_state_dict_or_path
        model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())
        prefix = "module."

        toload = {}
        mismatched_shape_keys = []
        # loaded_not_model_keys = []
        not_loaded_model_keys = []
        for k in model_keys:
            if k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[k]
            elif prefix + k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[prefix +
                                                                   k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[prefix + k]
            else:
                not_loaded_model_keys.append(k)

        # LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
        # LOGGER.info("Keys in loaded but not in model:")
        # diff_keys = load_keys.difference(model_keys)
        # LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
        print("Keys in model but not in loaded:")
        # diff_keys = model_keys.difference(load_keys)
        print(
            f"In total {len(not_loaded_model_keys)}, {sorted(not_loaded_model_keys)}"
        )
        print("Keys in model and loaded, but shape mismatched:")
        print(
            f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}"
        )
        self.load_state_dict(toload, strict=False)


class ObjectQARelation(BaseModel):

    def __init__(self,
                 object_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='bilinear'):
        super().__init__()
        self.text_params = text_params
        self.object_params = object_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError(
                "Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        self.object_model = load_clip_pt_weight(
            ObjectTransformer(input_dim=2054,
                              region_nums=self.object_params['object_num'],
                              output_dim=256,
                              time_module=self.object_params['time_module'],
                              num_frames=self.object_params['num_frames']))
        self.head = BUTDQAHead(v_dim=256,
                               q_dim=256,
                               hid_dim=256,
                               out_dim=self.object_params['num_label'])
        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim),
            )
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint,
                                    map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(
                state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            try:
                self.load_state_dict(new_state_dict, strict=True)
            except Exception as e:
                print("Parameters of model and state_dict are mismatched. {}".
                      format(e))
                self.load_state_dict_with_mismatch(new_state_dict)
        self.segments = self.object_params['num_frames']
        self.projection_dim = 256

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        text_mask = text_data['attention_mask']
        object_data = data['object']
        object_mask = data['object_mask']
        text_embeddings = self.compute_text(text_data)
        object_embeddings, _ = self.compute_object(object_data, object_mask)
        logits = self.compute_fusion(text_embeddings, object_embeddings,
                                     text_mask, object_mask)
        return dict(logits=logits)

    def compute_text(self, text_data, pad=False):
        text_embeddings_all = self.text_model(**text_data).last_hidden_state
        text_embeddings = self.txt_proj(text_embeddings_all)
        return text_embeddings

    def compute_object(self, object_data, object_mask):
        object_embeddings, object_mask = self.object_model(
            object_data, object_mask)
        return object_embeddings, object_mask


    def compute_fusion(self, text_embeddings, object_embeddings, text_mask,
                       object_mask):
        if len(object_mask.shape) == 3:
            object_mask = object_mask.view(
                -1, self.object_params['num_frames'] *
                self.object_params['object_num']).type_as(object_embeddings)
        text_embeddings, _ = torch.max(text_embeddings, dim=1)
        logits = self.head(text_embeddings, object_embeddings[:, 1:],
                           object_mask)
        return logits

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'object_model.temporal_embed' in new_state_dict and 'object_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['object_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.object_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(
                        f'### loaded {self.object_params["model"]} model has MORE frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    new_temporal_embed = load_temporal_embed[:, :
                                                             curr_num_frames, :]
                else:
                    print(
                        f'### loaded {self.object_params["model"]} model has FEWER frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([
                            load_temporal_embed.shape[0], curr_num_frames,
                            embed_dim
                        ])
                        new_temporal_embed[:, :
                                           load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed, (curr_num_frames, embed_dim),
                            mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict[
                    'object_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'object_model.custom_pos_embed' in new_state_dict and 'object_model.custom_pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['object_model.custom_pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['object_model.custom_pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.'
                )
        return new_state_dict

    def load_state_dict_with_mismatch(self, loaded_state_dict_or_path):
        """operated in-place, no need to return `model`"""

        if isinstance(loaded_state_dict_or_path, str):
            loaded_state_dict = torch.load(loaded_state_dict_or_path,
                                           map_location="cpu")
        else:
            loaded_state_dict = loaded_state_dict_or_path
        model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())
        prefix = "module."

        toload = {}
        mismatched_shape_keys = []
        # loaded_not_model_keys = []
        not_loaded_model_keys = []
        for k in model_keys:
            if k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[k]
            elif prefix + k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[prefix +
                                                                   k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[prefix + k]
            else:
                not_loaded_model_keys.append(k)

        # LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
        # LOGGER.info("Keys in loaded but not in model:")
        # diff_keys = load_keys.difference(model_keys)
        # LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
        print("Keys in model but not in loaded:")
        # diff_keys = model_keys.difference(load_keys)
        print(
            f"In total {len(not_loaded_model_keys)}, {sorted(not_loaded_model_keys)}"
        )
        print("Keys in model and loaded, but shape mismatched:")
        print(
            f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}"
        )
        self.load_state_dict(toload, strict=False)


class ObjectMCRelation(BaseModel):

    def __init__(self,
                 object_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()
        self.text_params = text_params
        self.object_params = object_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError(
                "Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        self.object_model = load_clip_pt_weight(
            ObjectTransformer(input_dim=2054,
                              region_nums=self.object_params['object_num'],
                              output_dim=256,
                              time_module=self.object_params['time_module'],
                              num_frames=self.object_params['num_frames']))

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim),
            )
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint,
                                    map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(
                state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            try:
                self.load_state_dict(new_state_dict, strict=True)
                print("Loaded state_dict...")
            except Exception as e:
                print("Parameters of model and state_dict are mismatched. {}".
                      format(e))
                self.load_state_dict_with_mismatch(new_state_dict)

        self.segments = self.object_params['num_frames']
        self.projection_dim = 256

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        object_data = data['object']
        object_mask = data['object_mask']
        global_text_embeddings, local_text_embeddings = self.compute_text(
            text_data)
        global_object_embeddings, local_object_embeddings, object_mask = self.compute_object(
            object_data, object_mask)
        return dict(
            global_text_embeddings=global_text_embeddings.contiguous(),
            local_text_embeddings=local_text_embeddings.contiguous(),
            global_object_embeddings=global_object_embeddings.contiguous(),
            local_object_embeddings=local_object_embeddings.contiguous(),
            object_mask=object_mask[:, 1:, ...].contiguous(),
        )

    def compute_text(self, text_data, pad=False):
        text_embeddings_all = self.text_model(**text_data).last_hidden_state
        text_embeddings = self.txt_proj(text_embeddings_all)
        global_text_embeddings, local_text_embeddings = text_embeddings[:, 0,...], text_embeddings[:,1:,...]
        return global_text_embeddings, local_text_embeddings


    def compute_object(self, object_data, object_mask):
        object_embeddings, object_mask = self.object_model(
            object_data, object_mask)
        global_object_embeddings, local_object_embeddings = object_embeddings[:,0,...], object_embeddings[:,1:,...]
        return global_object_embeddings, local_object_embeddings, object_mask


    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'object_model.temporal_embed' in new_state_dict and 'object_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['object_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.object_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(
                        f'### loaded {self.object_params["model"]} model has MORE frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    new_temporal_embed = load_temporal_embed[:, :
                                                             curr_num_frames, :]
                else:
                    print(
                        f'### loaded {self.object_params["model"]} model has FEWER frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([
                            load_temporal_embed.shape[0], curr_num_frames,
                            embed_dim
                        ])
                        new_temporal_embed[:, :
                                           load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed, (curr_num_frames, embed_dim),
                            mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict[
                    'object_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'object_model.custom_pos_embed' in new_state_dict and 'object_model.custom_pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['object_model.custom_pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['object_model.custom_pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.'
                )
        return new_state_dict

    def load_state_dict_with_mismatch(self, loaded_state_dict_or_path):
        """operated in-place, no need to return `model`"""

        if isinstance(loaded_state_dict_or_path, str):
            loaded_state_dict = torch.load(loaded_state_dict_or_path,
                                           map_location="cpu")
        else:
            loaded_state_dict = loaded_state_dict_or_path
        model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())
        prefix = "module."

        toload = {}
        mismatched_shape_keys = []
        # loaded_not_model_keys = []
        not_loaded_model_keys = []
        for k in model_keys:
            if k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[k]
            elif prefix + k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[prefix +
                                                                   k].shape:
                    mismatched_shape_keys.append(k)
                else:
                    toload[k] = loaded_state_dict[prefix + k]
            else:
                not_loaded_model_keys.append(k)

        # LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
        # LOGGER.info("Keys in loaded but not in model:")
        # diff_keys = load_keys.difference(model_keys)
        # LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
        print("Keys in model but not in loaded:")
        # diff_keys = model_keys.difference(load_keys)
        print(
            f"In total {len(not_loaded_model_keys)}, {sorted(not_loaded_model_keys)}"
        )
        print("Keys in model and loaded, but shape mismatched:")
        print(
            f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}"
        )
        self.load_state_dict(toload, strict=False)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    pass
