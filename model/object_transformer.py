import torch.nn as nn
import torch
from collections import OrderedDict

from functools import partial
from collections import OrderedDict
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x, x_mask):
        attn_mask = x_mask.to(dtype=x.dtype,
                              device=x.device) if x_mask is not None else None
        x = x.transpose(1, 0)
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=attn_mask)[0].transpose(1, 0)

    def forward(self, x, x_mask):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


def attn_mask(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if mask is not None:
        sim += mask
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_mask, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))
        x_mask = torch.repeat_interleave(x_mask, h, dim=0)

        q *= self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_), (cls_mask, mask) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v, x_mask))
        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn_mask(cls_q, k, v, x_mask.unsqueeze(1))
        # rearrange across time or space
        q_, k_, v_, mask = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **
                                einops_dims), (q_, k_, v_, mask.unsqueeze(-1)))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v, cls_mask = map(
            lambda t: repeat(t, 'b () d -> (b r) () d', r=r),
            (cls_k, cls_v, cls_mask.unsqueeze(-1)))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)
        x_mask = torch.cat((cls_mask, mask), dim=1).squeeze()

        # attention
        out = attn_mask(q_, k_, v_, x_mask.unsqueeze(1))

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class SpaceTimeBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 time_init='zeros',
                 attention_style='frozen-in-time',
                 time_module=None,
                 num_frames=4):
        super().__init__()
        # print(dim)
        self.norm1 = norm_layer(dim)
        self.time_module = time_module
        self.attn = VarAttention(dim,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=drop)

        if self.time_module == 'timeattn':
            self.timeattn = VarAttention(dim,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         attn_drop=attn_drop,
                                         proj_drop=drop,
                                         initialize=time_init)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style

    def forward(self, x, x_mask, einops_from_space, einops_to_space,
                einops_from_time, einops_to_time, time_n, space_f):

        if self.time_module == 'timeattn':
            time_output = self.timeattn(self.norm3(x),
                                        x_mask,
                                        einops_from_time,
                                        einops_to_time,
                                        n=time_n)
            time_residual = x + time_output
        else:
            time_residual = x
        space_output = self.attn(self.norm1(time_residual),
                                 x_mask,
                                 einops_from_space,
                                 einops_to_space,
                                 f=space_f)
        if self.attention_style == 'frozen-in-time':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        x = space_residual + self.drop_path(
            self.mlp(self.norm2(space_residual)))

        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ObjectTransformer(nn.Module):

    def __init__(self,
                 input_dim=2054,
                 region_nums=20,
                 num_frames=4,
                 output_dim=256,
                 time_module=None):
        super().__init__()
        in_chans = 3
        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        representation_size = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        hybrid_backbone = None
        num_frames = num_frames
        time_init = 'rand'
        attention_style = 'frozen-in-time'

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = region_nums * num_frames
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.custom_pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1, embed_dim)
        )  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(dim=embed_dim,
                           num_heads=num_heads,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias,
                           qk_scale=qk_scale,
                           drop=drop_rate,
                           attn_drop=attn_drop_rate,
                           drop_path=dpr[i],
                           norm_layer=norm_layer,
                           time_init=time_init,
                           attention_style=attention_style,
                           time_module=time_module,
                           num_frames=num_frames) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(embed_dim, representation_size)),
                             ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        trunc_normal_(self.custom_pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)
        ## einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'
        #
        self.feat_dim = 2048
        self.red_dim = 768
        self.object_embedding = nn.Linear(self.feat_dim, self.embed_dim)
        self.pos_embedding = nn.Linear(input_dim - self.feat_dim,
                                       self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, output_dim, bias=False)
        #

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def forward_features(self, x, x_mask):
        # print(x.size()) # [batchsize*segments, 1, topk, 2054]
        b, curr_frames, L, C = x.shape
        _, _, object_num = x_mask.shape
        object_feature = x[:, :, :, :self.feat_dim]
        position_feature = x[:, :, :, self.feat_dim:]
        position_feature = self.pos_embedding(position_feature)
        x = self.object_embedding(object_feature)
        x += position_feature
        x = x.reshape(b, -1, self.embed_dim)  # [batchsize, segments*topk, 768]
        x_mask = x_mask.reshape(b, -1)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(
            BF, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks [batchsize, 1, 768]
        cls_mask = torch.ones(BF, 1).type_as(x_mask)  # [batchsize, 1]
        x = torch.cat((cls_tokens, x),
                      dim=1)  #[batchsize, segments*topk+1, 768]
        x_mask = torch.cat((cls_mask, x_mask),
                           dim=1)  # [BF, segments*topk + 1]
        x_mask = (x_mask - 1) * 100  # change mask to [0, 0, 0, -100, -100...]
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.custom_pos_embed[:, 0, :].unsqueeze(1)  #[1, 1, 768]
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(
            self.patches_per_frame, 1)  #[1, segments*topk, 768]
        total_pos_embed = tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed],
                                    dim=1)  #[1, segments*topk + 1, 2048]

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for blk in self.blocks:
            x = blk(x,
                    x_mask,
                    self.einops_from_space,
                    self.einops_to_space,
                    self.einops_from_time,
                    self.einops_to_time,
                    time_n=n,
                    space_f=f)
        x = self.pre_logits(x)
        return x, x_mask

    def forward(self, x, x_mask):
        x, x_mask = self.forward_features(x, x_mask)
        x = self.proj(x)
        return x, x_mask


def weight_transform(model_dict, pretrain_dict):
    '''
    :return:
    '''
    weight_dict = {
        k[7:]: v
        for k, v in pretrain_dict.items()
        if k[7:] in model_dict and k[:7] == 'visual.'
    }
    for k, v in weight_dict.items():
        print("load: {}".format(k))
    model_dict.update(weight_dict)
    return model_dict


def load_clip_pt_weight(model):
    """
    load the object transformer weight from clip vision transformer
    notice some of have failed
    Args:
        model ():

    Returns:

    """
    vit_checkpoint = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth",
                                map_location="cpu")
    model.load_state_dict(vit_checkpoint, strict=False)
    return model


if __name__ == '__main__':
    x = torch.zeros([2, 4, 100, 2054])
    x_mask = torch.ones([2, 4, 100])
    object_transformer = ObjectTransformer(2054, 100, 256)
    object_transformer = load_clip_pt_weight(object_transformer)
    object_transformer.eval()
    y, y_mask = object_transformer(x, x_mask)
    print(y.size(), y_mask.shape)