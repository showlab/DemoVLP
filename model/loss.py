import random
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch
import time
import numpy as np


class GlobalLocalLoss(nn.Module):

    def __init__(self,
                 temperature=0.05,
                 lambda_softmax=20,
                 focal_type="prob",
                 margin=0,
                 max_violation=False,
                 use_local=True,
                 use_global=True,
                 coef=1000.):
        super(GlobalLocalLoss, self).__init__()
        self.global_loss = NormSoftmaxLoss(temperature)
        self.local_loss = RWALoss(lambda_softmax, focal_type, margin,
                                  max_violation)
        self.use_local = use_local
        self.use_global = use_global
        self.cof_local = coef

    def forward(self, global_sim, local_im, local_s, local_im_m, local_s_l,
                local_s_m):
        if not self.use_local:
            loss = self.global_loss(global_sim)
            local_loss = torch.tensor([0.0])
            global_loss = loss
        elif not self.use_global:
            loss = self.local_loss(local_im, local_s, local_im_m, local_s_l,
                                   local_s_m)
            local_loss = loss
            global_loss = torch.tensor([0.0])
        else:
            global_loss = self.global_loss(global_sim)
            local_loss = self.local_loss(local_im, local_s, local_im_m,
                                         local_s_l, local_s_m)
            loss = global_loss + local_loss
        return loss, global_loss, local_loss


class RWALoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self,
                 lambda_softmax=20,
                 focal_type="prob",
                 margin=0,
                 max_violation=False):
        super(RWALoss, self).__init__()
        self.lambda_softmax = lambda_softmax
        self.focal_type = focal_type
        self.margin = margin
        self.max_violation = max_violation

    def get_sim(self, im, s, im_m, s_l, s_m):
        return xattn_score_fast(im,
                                s,
                                im_m,
                                s_l,
                                s_m,
                                lambda_softmax=self.lambda_softmax,
                                focal_type=self.focal_type)

    def get_sim_by_segment(self,
                           img_feats,
                           lang_feats,
                           img_mask,
                           lang_length,
                           cap_mask,
                           segment=8,
                           device="cpu"):
        """
        Used when testing, because length of features is too long
        """
        img_length = img_feats.shape[0]
        text_length = lang_feats.shape[0]
        n_im_shard = int((img_length - 1) / segment + 1)
        n_cap_shard = int((text_length - 1) / segment + 1)
        sim = np.zeros((img_length, text_length))
        for i in range(n_im_shard):
            im_start, im_end = segment * i, min(segment * (i + 1), img_length)
            for j in range(n_cap_shard):
                cap_start, cap_end = segment * j, min(segment * (j + 1),
                                                      text_length)
                im = img_feats[im_start:im_end].to(device)
                la = lang_feats[cap_start:cap_end].to(device)
                imm = img_mask[im_start:im_end].to(device)
                lal = lang_length[cap_start:cap_end].to(device)
                lam = cap_mask[cap_start:cap_end].to(device)
                o2t_sim = self.get_sim(im, la, imm, lal, lam)
                del im, la, imm, lal
                sim[im_start:im_end,
                    cap_start:cap_end] = o2t_sim.detach().cpu().numpy()
        return sim

    def forward(self, im, s, im_m, s_l, s_m):
        # compute image-sentence score matrix
        scores = self.get_sim(im, s, im_m, s_l, s_m)
        labels = torch.eye(im.shape[0]).type_as(scores)

        pred = F.softmax(scores * self.lambda_softmax, dim=1)
        loss = pred * (F.log_softmax(scores * self.lambda_softmax, dim=1) -
                       torch.log(labels + 1e-6))

        loss = torch.mean(torch.sum(loss, dim=1))

        return loss


class NormSoftmaxLoss(nn.Module):

    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return -loss_i - loss_j


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2),
                                         0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) -
           im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def nll_loss(output, target):
    return F.nll_loss(output, target)


def func_attention_fast(query,
                        context,
                        query_mask,
                        context_mask,
                        lambda_softmax=20,
                        focal_type='prob',
                        eps=1e-8):
    """
    query: (batchq, queryL, d)
    context: (batchc, sourceL, d)
    query_maks: (batchq, queryL, 1)
    context: (batchc, 1, sourceL)
    opt: parameters
    """
    batch_size_c, batch_size_q, queryL, sourceL = context.size(0), query.size(
        0), query.size(1), context.size(1)
    query = l2norm(query, dim=-1)
    context = l2norm(context, dim=-1)

    # Step 1: preassign attention
    # --> (batchq, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # --> (batchc, 1, sourceL, d)
    context = context.unsqueeze(1)
    # (batchc, 1, sourceL, d)(batchq, d, queryL)
    attn = torch.matmul(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    # (batchc, batchq, sourceL, queryL)
    attn = l2norm(attn, 3)

    # --> (batchc, batchq, queryL, sourceL)
    attn = torch.transpose(attn, 2, 3).contiguous()
    attn = attn + query_mask.unsqueeze(0) + context_mask.unsqueeze(1)
    # --> (batchc*batchq*queryL, sourceL)
    attn = attn.view(batch_size_c * batch_size_q * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * lambda_softmax)
    # --> (batchc, batchq, queryL, sourceL)
    attn = attn.view(batch_size_c, batch_size_q, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if focal_type == 'equal':
        funcH = focal_equal(attn, batch_size_c, queryL, sourceL)
    else:
        funcH = 1.0

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (batchc, 1, d, sourceL)
    contextT = torch.transpose(context, 2, 3)
    # --> (batchc, batchq, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 2, 3).contiguous().type_as(contextT)
    # (batchc x 1 x d x sourceL)(batchc x batchq x sourceL x queryL)
    # --> (batchc, batchq, d, queryL)
    weightedContext = torch.matmul(contextT, re_attnT)
    # --> (batchc, batchq, queryL, d)
    weightedContext = torch.transpose(weightedContext, 2, 3)

    return weightedContext


def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_fast(images,
                     captions,
                     img_mask,
                     cap_lens=None,
                     cap_mask=None,
                     focal_type='prob',
                     lambda_softmax=20):
    """
    Images: (n_image, n_regions, d) matrix of images
    ImgMask: (n_image, n_regions) array of region lengths
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapMask: (n_caption, n_word) array of caption lengths
    """
    if cap_mask == None:
        cap_mask = torch.ones(*captions.shape[:2]).type_as(img_mask)

    # (n_image, n_caption, n_word, d)
    weiContext = func_attention_fast(captions,
                                     images,
                                     cap_mask.unsqueeze(-1),
                                     img_mask.unsqueeze(1),
                                     lambda_softmax=lambda_softmax,
                                     focal_type=focal_type)
    i2t_sim = cosine_similarity(captions.unsqueeze(0), weiContext, dim=-1)
    i2t_sim = i2t_sim.mean(dim=-1)

    weiContext = func_attention_fast(images,
                                     captions,
                                     img_mask.unsqueeze(-1),
                                     cap_mask.unsqueeze(1),
                                     lambda_softmax=lambda_softmax,
                                     focal_type=focal_type)
    t2i_sim = cosine_similarity(images.unsqueeze(0), weiContext, dim=-1)
    t2i_sim = t2i_sim.mean(dim=-1)
    # (n_image, n_caption)
    sim = t2i_sim.T + i2t_sim
    return sim


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


if __name__ == "__main__":
    import torch

    random_sims = (torch.rand([10, 8]) * 2) - 1
    loss = NormSoftmaxLoss()
    loss(random_sims)

    loss = GlobalLocalLoss()
    random_sims = (torch.rand([128, 128]) * 2) - 1
    l_im = torch.randn(128, 400, 256)
    l_s = torch.randn(128, 100, 256)
    l_im_m = torch.ones(128, 400)
    l_s_l = torch.randint(low=2, high=100, size=(128, ))
    l = loss(random_sims, l_im, l_s, l_im_m, l_s_l)
    print(l)
