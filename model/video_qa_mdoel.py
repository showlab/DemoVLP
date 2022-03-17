"""
Refer to https://github.com/BierOne/bottom-up-attention-vqa
"""
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """
    Simple class for multi-layer non-linear fully connect network
    Activate function: ReLU()
    """

    def __init__(self, dims, dropout=0.0, norm=True):
        super(FCNet, self).__init__()
        self.num_layers = len(dims) - 1
        self.drop = dropout
        self.norm = norm
        self.main = nn.Sequential(*self._init_layers(dims))

    def _init_layers(self, dims):
        layers = []
        for i in range(self.num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            # layers.append(nn.Dropout(self.drop))
            if self.norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim),
                                          dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        return self.main(x)


class SimpleClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super(SimpleClassifier, self).__init__()
        self.q_net = FCNet([in_dim[0], hid_dim[0]], dropout)
        self.v_net = FCNet([in_dim[1], hid_dim[0]], dropout)
        self.main = nn.Sequential(nn.Linear(hid_dim[0], hid_dim[1]), nn.ReLU(),
                                  nn.Dropout(dropout, inplace=True),
                                  nn.Linear(hid_dim[1], out_dim))

    def forward(self, q_emb, v_emb):
        joint_repr = self.q_net(q_emb) * self.v_net(v_emb)
        logits = self.main(joint_repr)
        return logits


class Attention(nn.Module):

    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)

    def forward(self, v, v_mask, q):
        """
        v: [batch, k, vdim]
        v_mask: [batch, k]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, hid_dim]
        q_proj = self.q_proj(q).unsqueeze(1)  # [batch, 1, hid_dim]
        logits = self.linear(self.drop(v_proj * q_proj))  # [batch, k, 1]
        logits = logits * v_mask.unsqueeze(-1)
        return nn.functional.softmax(logits, 1), logits


class BUTDQAHead(nn.Module):

    def __init__(self, v_dim, q_dim, hid_dim, out_dim):
        super(BUTDQAHead, self).__init__()
        self.v_att = Attention(v_dim, q_dim, hid_dim)
        self.classifier = SimpleClassifier([q_dim, v_dim],
                                           [hid_dim, hid_dim * 2], out_dim)

    def forward(self, txt_embed, obj_embed, obj_mask):
        """Forward
        txt_embed: [batch, txt_dim]
        obj_embed: [batch, num_objs, obj_dim]
        obj_mask: [batch_size, num_objs]
        return: logits, attention_weights
        """
        att, att_logits = self.v_att(obj_embed, obj_mask,
                                     txt_embed)  # [batch, objs, 1]
        obj_embed = (att * obj_embed).sum(1)  # [batch, v_dim]
        logits = self.classifier(txt_embed, obj_embed)
        return logits
