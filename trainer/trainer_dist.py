import os
import numpy as np
import torch
from torch import nn
from base.base_trainer import Multi_BaseTrainer_dist
from utils import inf_loop, flat_list_of_lists, save_json, load_json, merge_dicts
from model.model import sim_matrix
import torch.distributed as dist
from torch import nn
import time


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank:ctx.batch_size *
                        (ctx.rank + 1)],
            None,
            None,
        )


# for distributed train
class Multi_ObjectTrainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self,
                 args,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 config,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 len_epoch=None,
                 writer=None,
                 visualizer=None,
                 tokenizer=None,
                 max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__),
                                       acc_metrics[i])
        return acc_metrics

    def _pseudo_label_loss(self, predict, gt):
        loss_f = nn.BCELoss()
        return loss_f(predict, gt.type(torch.float))

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx +
                    1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                begin_time = time.time()
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'],
                                                  return_tensors='pt',
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  max_length=100)
                data['text'] = {
                    key: val.to(self.device)
                    for key, val in data['text'].items()
                }
                data['object'] = data['object'].to(self.device)
                data['object_mask'] = data['object_mask'].to(self.device)
                text_length = torch.sum(data['text']['attention_mask'], dim=1)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    output_dict = self.model(data)
                    global_text_embeds = output_dict['global_text_embeddings']
                    global_object_embeds = output_dict[
                        'global_object_embeddings']
                    local_text_embeds = output_dict['local_text_embeddings']
                    local_object_embeds = output_dict[
                        'local_object_embeddings']
                    object_mask = output_dict['object_mask']
                    text_mask = data['text']['attention_mask'][:,
                                                               1:].contiguous(
                                                               )
                    text_mask = (text_mask - 1.0) * 100.0
                    global_sim = sim_matrix(global_text_embeds,
                                            global_object_embeds)
                    loss, global_loss, local_loss = self.loss(
                        global_sim, local_object_embeds, local_text_embeds,
                        object_mask, text_length, text_mask)  # normal t2v loss
                loss.backward()
                end_time = time.time()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    print("loss:{}, global_loss: {}, local_loss: {}".format(
                        loss.item(), global_loss.item(), local_loss.item()))
                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}',
                                           loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug(
                        'Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                            epoch, dl_idx, self._progress(batch_idx, dl_idx),
                            loss.detach().item()))

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch
            for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation and (epoch % 1) == 0:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        # if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(
            self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        object_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        local_text_embed_arr = {
            x: []
            for x in range(len(self.valid_data_loader))
        }
        local_object_embed_arr = {
            x: []
            for x in range(len(self.valid_data_loader))
        }
        text_length_arr = {x: [] for x in range(len(self.valid_data_loader))}
        object_mask_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_mask_arr = {x: [] for x in range(len(self.valid_data_loader))}
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'],
                                                      return_tensors='pt',
                                                      pad_to_max_length=True,
                                                      truncation=True,
                                                      max_length=100)
                    data['text'] = {
                        key: val.to(self.device)
                        for key, val in data['text'].items()
                    }
                    data['object'] = data['object'].to(self.device)
                    data['object_mask'] = data['object_mask'].to(self.device)
                    text_length = torch.sum(data['text']['attention_mask'],
                                            dim=1)
                    text_length_all = [
                        torch.zeros_like(text_length)
                        for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(text_length_all, text_length)
                    text_length_all = torch.cat(text_length_all, dim=0)

                    text_mask_all = [
                        torch.zeros_like(
                            data['text']['attention_mask'][:, 1:].contiguous())
                        for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(
                        text_mask_all,
                        data['text']['attention_mask'][:, 1:].contiguous())
                    text_mask_all = torch.cat(text_mask_all, dim=0)
                    output_dict = self.model.module(data, return_embeds=True)
                    # output_dict = self.model(data, return_embeds=True)
                    if output_dict['global_text_embeddings'] is not None:
                        global_text_embed_all = [
                            torch.zeros_like(
                                output_dict['global_text_embeddings'])
                            for _ in range(self.n_gpu)
                        ]
                        torch.distributed.all_gather(
                            global_text_embed_all,
                            output_dict['global_text_embeddings'])
                        global_text_embed_all = torch.cat(
                            global_text_embed_all, dim=0)
                    if output_dict['global_object_embeddings'] is not None:
                        global_object_embed_all = [
                            torch.zeros_like(
                                output_dict['global_object_embeddings'])
                            for _ in range(self.n_gpu)
                        ]
                        torch.distributed.all_gather(
                            global_object_embed_all,
                            output_dict['global_object_embeddings'])
                        global_object_embed_all = torch.cat(
                            global_object_embed_all, dim=0)
                    if output_dict['local_text_embeddings'] is not None:
                        local_text_embed_all = [
                            torch.zeros_like(
                                output_dict['local_text_embeddings'])
                            for _ in range(self.n_gpu)
                        ]
                        torch.distributed.all_gather(
                            local_text_embed_all,
                            output_dict['local_text_embeddings'])
                        local_text_embed_all = torch.cat(local_text_embed_all,
                                                         dim=0)
                    if output_dict['local_object_embeddings'] is not None:
                        local_object_embed_all = [
                            torch.zeros_like(
                                output_dict['local_object_embeddings'])
                            for _ in range(self.n_gpu)
                        ]
                        torch.distributed.all_gather(
                            local_object_embed_all,
                            output_dict['local_object_embeddings'])
                        local_object_embed_all = torch.cat(
                            local_object_embed_all, dim=0)
                    if output_dict['object_mask'] is not None:
                        object_mask_all = [
                            torch.zeros_like(output_dict['object_mask'])
                            for _ in range(self.n_gpu)
                        ]
                        torch.distributed.all_gather(
                            object_mask_all, output_dict['object_mask'])
                        object_mask_all = torch.cat(object_mask_all, dim=0)

                    text_mask_all = (text_mask_all - 1.0) * 100.0
                    text_embed_arr[dl_idx].append(global_text_embed_all.cpu())
                    object_embed_arr[dl_idx].append(
                        global_object_embed_all.cpu())
                    local_text_embed_arr[dl_idx].append(
                        local_text_embed_all.cpu())
                    local_object_embed_arr[dl_idx].append(
                        local_object_embed_all.cpu())
                    text_length_arr[dl_idx].append(text_length_all.cpu())
                    object_mask_arr[dl_idx].append(object_mask_all.cpu())
                    text_mask_arr[dl_idx].append(text_mask_all.cpu())

                    global_sims_batch = sim_matrix(global_text_embed_all,
                                                   global_object_embed_all)
                    loss, global_loss, local_loss = self.loss(
                        global_sims_batch, local_object_embed_all,
                        local_text_embed_all, object_mask_all, text_length_all,
                        text_mask_all)  # normal video to text loss
                    if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                        print(
                            "loss:{}, global_loss: {}, local_loss: {}".format(
                                loss.item(), global_loss.item(),
                                local_loss.item()))
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(
                    f'loss_val_{dl_idx}', total_val_loss[dl_idx] /
                    len(self.valid_data_loader[dl_idx]))
            nested_metrics = {
                x: {}
                for x in range(len(self.valid_data_loader))
            }
            text_embeds = torch.cat(text_embed_arr[dl_idx])
            object_embeds = torch.cat(object_embed_arr[dl_idx])
            local_text_embed = torch.cat(local_text_embed_arr[dl_idx])
            local_object_embed = torch.cat(local_object_embed_arr[dl_idx])
            text_length = torch.cat(text_length_arr[dl_idx])
            object_mask = torch.cat(object_mask_arr[dl_idx])
            text_mask = torch.cat(text_mask_arr[dl_idx])
            if self.config["name"].startswith("MSCOCO"):
                object_embeds = object_embeds[::5, ...]
                local_object_embed = local_object_embed[::5, ...]
                object_mask = object_mask[::5, ...]
            o2t_sims = sim_matrix(text_embeds,
                                  object_embeds).detach().cpu().numpy()
            if self.config["loss"]["args"]["use_local"] is True:
                if self.args.local_rank == 0:
                    print("Global similarity: ", o2t_sims[0][:10])
                    print("Start to compute local similarity...")
                local_o2t_sims = self.loss.local_loss.get_sim_by_segment(
                    local_object_embed,
                    local_text_embed,
                    object_mask,
                    text_length,
                    text_mask,
                    device=self.device)
                if self.args.local_rank == 0:
                    print("Local similarity: ", local_o2t_sims[0][:10])
                    print("End compute local similarity...")
                o2t_sims = o2t_sims + local_o2t_sims
            for metric in self.metrics:
                metric_name = metric.__name__
                if self.config["name"].startswith("MSCOCO"):
                    res = metric(o2t_sims, fold=5)
                else:
                    res = metric(o2t_sims)
                if self.args.rank == 0:
                    print("object to caption")
                if self.args.rank == 0:
                    verbose(epoch=epoch,
                            metrics=res,
                            name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res
        del text_embed_arr, object_embed_arr, local_text_embed_arr, text_length_arr, object_mask_arr
        res_dict = {
            f'val_loss_{dl_idx}':
            total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
            for dl_idx in range(len(self.valid_data_loader))
        }
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Multi_ObjectQATrainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self,
                 args,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 config,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 len_epoch=None,
                 writer=None,
                 visualizer=None,
                 tokenizer=None,
                 max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

        self.valid_qid2data = {}
        self.valid_label2ans = {}
        for dl_idx, dl in enumerate(self.valid_data_loader):
            self.valid_label2ans[dl_idx] = dl.dataset.label2ans
            self.valid_qid2data[dl_idx] = dl.dataset.qid2data

    def _pseudo_label_loss(self, predict, gt):
        loss_f = nn.BCELoss()
        return loss_f(predict, gt.type(torch.float))

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        pos_cnt = 0.
        tot_cnt = 0.
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx +
                    1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                begin_time = time.time()
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'],
                                                  return_tensors='pt',
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  max_length=100)
                data['text'] = {
                    key: val.to(self.device)
                    for key, val in data['text'].items()
                }
                data['object'] = data['object'].to(self.device)
                data['object_mask'] = data['object_mask'].to(self.device)
                text_length = torch.sum(data['text']['attention_mask'], dim=1)
                label = data['label'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    output_dict = self.model(data)
                    logits = output_dict['logits']
                    loss = self.loss(logits, label)
                    # compute acc
                    pred = logits.max(dim=-1)[1]
                    pos_cnt += (pred == label).long().sum().cpu().item()
                    tot_cnt += label.shape[0]
                loss.backward()
                end_time = time.time()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    print("loss:{}, acc: {}, postive/all : {}/{}".format(
                        loss.item(), pos_cnt / tot_cnt, pos_cnt, tot_cnt))
                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}',
                                           loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug(
                        'Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                            epoch, dl_idx, self._progress(batch_idx, dl_idx),
                            loss.detach().item()))

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch
            for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation and (epoch % 1) == 0:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        # if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(
            self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        logits_arr = {x: [] for x in range(len(self.valid_data_loader))}
        question_id_arr = {x: [] for x in range(len(self.valid_data_loader))}
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'],
                                                      return_tensors='pt',
                                                      pad_to_max_length=True,
                                                      truncation=True,
                                                      max_length=100)
                    data['text'] = {
                        key: val.to(self.device)
                        for key, val in data['text'].items()
                    }
                    data['object'] = data['object'].to(self.device)
                    data['object_mask'] = data['object_mask'].to(self.device)
                    text_length = torch.sum(data['text']['attention_mask'],
                                            dim=1)
                    label = data['label'].to(self.device)
                    qid = data['question_id'].to(self.device)
                    text_length_all = [
                        torch.zeros_like(text_length)
                        for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(text_length_all, text_length)
                    text_length_all = torch.cat(text_length_all, dim=0)
                    label_all = [
                        torch.zeros_like(label) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(label_all, label)
                    label_all = torch.cat(label_all, dim=0)
                    qid_all = [
                        torch.zeros_like(qid) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(qid_all, qid)
                    qid_all = torch.cat(qid_all, dim=0)
                    # output_dict = self.model.module(
                    #     data, return_embeds=True)
                    output_dict = self.model(data, return_embeds=True)
                    logits = output_dict['logits']
                    logits_all = [
                        torch.zeros_like(logits) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(logits_all, logits)
                    logits_all = torch.cat(logits_all, dim=0)

                    logits_arr[dl_idx].append(logits_all.cpu())
                    question_id_arr[dl_idx].append(qid_all.cpu())

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(
                    f'loss_val_{dl_idx}', total_val_loss[dl_idx] /
                    len(self.valid_data_loader[dl_idx]))

            assert len(logits_arr) == len(question_id_arr)
            results = []
            for dl_idx in range(len(logits_arr)):
                assert len(logits_arr[dl_idx]) == len(question_id_arr[dl_idx])
                for idx in range(len(logits_arr[dl_idx])):
                    pred_label = logits_arr[dl_idx][idx].max(
                        dim=-1)[1].data.tolist()
                    qid = question_id_arr[dl_idx][idx].data.tolist()
                    for q, pred in zip(qid, pred_label):
                        results.append(
                            dict(question_id=q,
                                 answer=pred,
                                 data=self.valid_qid2data[dl_idx][q]))
            print(f"Get {len(results)} results.")
            nested_metrics = {
                x: {}
                for x in range(len(self.valid_data_loader))
            }
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(results, self.valid_label2ans[dl_idx],
                             self.valid_qid2data[dl_idx])
                if self.args.rank == 0:
                    print("object to caption")
                if self.args.rank == 0:
                    print(res)
                    # verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                    #         mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res
        res_dict = {
            f'val_loss_{dl_idx}':
            total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
            for dl_idx in range(len(self.valid_data_loader))
        }
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Multi_ObjectMCTrainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self,
                 args,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 config,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 len_epoch=None,
                 writer=None,
                 visualizer=None,
                 tokenizer=None,
                 max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

        self.valid_gt_id2answer = {}
        for dl_idx, dl in enumerate(self.valid_data_loader):
            self.valid_gt_id2answer[dl_idx] = dl.dataset.id2answer

    def _pseudo_label_loss(self, predict, gt):
        loss_f = nn.BCELoss()
        return loss_f(predict, gt.type(torch.float))

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        return None

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(
            self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_id2answer = {
            x: dict()
            for x in range(len(self.valid_data_loader))
        }
        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = flat_list_of_lists(data['text'])
                        data['text'] = self.tokenizer(data['text'],
                                                      return_tensors='pt',
                                                      pad_to_max_length=True,
                                                      truncation=True,
                                                      max_length=100)
                    data['text'] = {
                        key: val.to(self.device)
                        for key, val in data['text'].items()
                    }
                    data['object'] = data['object'].to(self.device)
                    data['object_mask'] = data['object_mask'].to(self.device)
                    text_length = torch.sum(data['text']['attention_mask'],
                                            dim=1)
                    text_mask = data['text']['attention_mask'][:, 1:]
                    text_mask = (text_mask - 1.0) * 100.0
                    n_options = data['text']['attention_mask'].shape[0]
                    data['object'] = data['object'].expand(
                        n_options, -1, -1, -1)
                    data['object_mask'] = data['object_mask'].expand(
                        n_options, -1, -1)

                    output_dict = self.model.module(data, return_embeds=True)

                    global_sims_batch = sim_matrix(
                        output_dict['global_text_embeddings'],
                        output_dict['global_object_embeddings'])
                    local_sims_batch = self.loss.local_loss.get_sim(
                        output_dict['local_object_embeddings'],
                        output_dict['local_text_embeddings'],
                        output_dict['object_mask'], text_length, text_mask)
                    sims_batch = global_sims_batch + local_sims_batch
                    pred_answer = sims_batch[0].max(0)[1].tolist()
                    pred_id2answer[dl_idx][data['mc_id'][0]] = int(pred_answer)

        if self.n_gpu > 1:
            save_json(
                pred_id2answer,
                f"./tmp_results_mc_rank{self.args.local_rank}_dataset{dl_idx}.json"
            )
        # sync
        torch.distributed.barrier()
        if self.n_gpu > 1 and self.args.local_rank == 0:
            pred_id2ans = []

            for rk in range(self.n_gpu):
                pred_id2ans.append(
                    load_json(f"tmp_results_mc_rank{rk}_dataset{dl_idx}.json"))
                os.remove(f"tmp_results_mc_rank{rk}_dataset{dl_idx}.json")

            pred_id2ans = [i['0'] for i in pred_id2ans]
            pred_id2ans = merge_dicts(pred_id2ans)
        else:
            pred_id2ans = pred_id2answer[0]

        if self.args.local_rank == 0:
            for dl_idx in range(len(self.valid_data_loader)):
                # TODO: this needs a clean
                if self.writer is not None:
                    self.writer.log_scalar(
                        f'loss_val_{dl_idx}', total_val_loss[dl_idx] /
                        len(self.valid_data_loader[dl_idx]))
                nested_metrics = {
                    x: {}
                    for x in range(len(self.valid_data_loader))
                }
                for metric in self.metrics:
                    metric_name = metric.__name__
                    # print(pred_id2ans[str(dl_idx)])
                    # print(self.valid_gt_id2answer[dl_idx])
                    res = metric(pred_id2ans, self.valid_gt_id2answer[dl_idx])
                    if self.args.rank == 0:
                        print("object to caption")
                    if self.args.rank == 0:
                        print(res)
                        # verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                        #         mode=metric_name)
                    nested_metrics[dl_idx][metric_name] = res
            res_dict = {
                f'val_loss_{dl_idx}':
                total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                for dl_idx in range(len(self.valid_data_loader))
            }
            res_dict['nested_val_metrics'] = nested_metrics
        else:
            res_dict = None

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics[
        "R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
