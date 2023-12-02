import torch
import torch.nn as nn
import torch.nn.parallel

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import WBCELoss, PKDLoss, ContLoss
from data_loader import ADE

class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None, visulized_dir=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 100-50: 100 | 100-10: 100 | 50-50: 50 |

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        pos_weight = torch.ones(
            [len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
        self.BCELoss = WBCELoss(
            pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)

        self._print_train_info()

        self.visulized_dir = visulized_dir

        if not config['test']:
            self.compute_cls_number(self.config)

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features, _ = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_val.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{ADE[i]} {met()['by_class'][i]:.2f}\n"
                        elif i in self.evaluator_val.old_classes_idx:
                            by_class_str = by_class_str + f"{i:2d}  {ADE[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():

            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features, _ = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

                self.progress(self.logger, batch_idx, len(self.test_loader))

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{ADE[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {ADE[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)

        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_pkd', 'loss_cont',
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.BCELoss_fake = WBCELoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.BCELoss_extra_bg = WBCELoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.PKDLoss = PKDLoss()
        self.ContLoss = ContLoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)

        # self.pred_numbers = self.compute_pred_number()

        prev_info_path = \
            str(config.save_dir)[:-1] + \
            str(config['data_loader']['args']['task']['step'] - 1) + \
            "/prototypes-epoch{}.pth".format(config['trainer']['epochs'])

        prev_info = torch.load(prev_info_path)
        self.prev_numbers = prev_info['numbers'].to(self.device)
        self.prev_prototypes = prev_info['prototypes'].to(self.device)
        self.prev_norm = prev_info['norm_mean_and_std'].to(self.device)
        self.prev_noise = prev_info['noise'].to(self.device)

        self.current_numbers = self.numbers[1:].sum()
        # self.per_iter_prev_number = (self.prev_numbers[1:] / len(self.train_loader)).to(torch.int)

        assert task_info['setting'] in ['overlap', 'disjoint']

        if task_info['setting'] == 'overlap':
            self.prev_bg_number = self.prev_numbers[0] * (1 - 0.01 * self.n_new_classes)
        else:
            self.prev_bg_number = self.prev_numbers[0]

        # self.prev_numbers = self.prev_numbers[1:]

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + "
                         f"{self.config['hyperparameter']['pkd']} * L_pkd")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)

        if epoch == 1:
            self.pred_numbers = self.compute_pred_number()
            if self.task_info['setting'] == 'overlap':
                self.per_iter_prev_number = \
                    ((self.prev_numbers[1:] - self.pred_numbers[1:]) / len(self.train_loader)).to(torch.int)
            else:
                self.per_iter_prev_number = \
                    (self.prev_numbers[1:] / len(self.train_loader)).to(torch.int)

            self.per_iter_prev_number = torch.clamp(self.per_iter_prev_number, min=0)

            if self.task_info['setting'] == 'overlap':
                self.extra_bg_ratio = (self.prev_bg_number / self.pred_numbers[0]) - 0.5
            else:
                self.extra_bg_ratio = (self.prev_bg_number / self.pred_numbers[0])

            self.extra_bg_ratio = torch.clamp(self.extra_bg_ratio, min=0)

            tot_numbers = self.prev_numbers.clone().to(self.device)
            if self.task_info['setting'] == 'overlap':
                # 现在背景可能有一部分和过去背景重复，假设是50%
                # 过去背景中，有一部分是现在类别
                tot_numbers[0] = self.pred_numbers[0] * 0.5 + self.prev_bg_number
            else:
                # 现在背景和过去背景无重复
                # 过去背景中也不存在现在类别
                tot_numbers[0] = self.pred_numbers[0] + self.prev_bg_number
                # 现在图片中有一些新的过去类别
                tot_numbers[1:] = self.prev_numbers[1:].to(self.device) + self.pred_numbers[1:]
            tot_numbers = torch.cat((tot_numbers.to(self.device), self.numbers[1:].to(self.device)), dim=0)
            self.numbers = tot_numbers

        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old, _ = self.model_old(data['image'], ret_intermediate=True)

                        pred = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                        idx = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                        idx = idx.sum(dim=1)  # logit: [N, H, W]
                        pred[idx == 0] = 0  # set background (non-target class)
                        pseudo_label_region = torch.logical_and(
                            data['label'] == 0, pred > 0
                        ).unsqueeze(1)

                fake_features = []
                for cls in range(0, self.per_iter_prev_number.shape[0]):
                    per_cls_fake_features = \
                        self.prev_prototypes[cls].reshape(1, -1, 1, 1).repeat(1, 1, self.per_iter_prev_number[cls], 1)
                    noise = \
                        torch.randn_like(per_cls_fake_features) * self.prev_noise[cls].reshape(1, -1, 1, 1)
                    per_cls_fake_features = per_cls_fake_features + noise
                    rand_norm = \
                        torch.randn_like(per_cls_fake_features) * \
                        self.prev_norm[1, cls].reshape(1, 1, 1, 1) + \
                        self.prev_norm[0, cls].reshape(1, 1, 1, 1)
                    per_cls_fake_features = per_cls_fake_features * rand_norm
                    fake_features.append(per_cls_fake_features)

                fake_features = torch.cat(fake_features, dim=2)
                fake_label = torch.zeros(1, fake_features.shape[2], 1, requires_grad=False).to(self.device)

                region_bg = torch.logical_and(pred == 0, data['label'] == 0)[:, 8::16, 8::16]

                logit, features, extra = \
                    self.model(data['image'], ret_intermediate=True, fake_features=fake_features, region_bg=region_bg)
                logits_for_fake = extra[0]
                logits_for_extra_bg = extra[1]

                # [|Ct|]
                loss_mbce_ori = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])

                if logits_for_extra_bg is not None:
                    extra_bg_label = torch.zeros(1, logits_for_extra_bg.shape[2], 1, requires_grad=False).to(self.device)
                    loss_mbce_extra_bg = self.BCELoss_extra_bg(
                        logits_for_extra_bg[:, -self.n_new_classes:],
                        extra_bg_label,
                    ).mean(dim=[0, 2, 3])
                else:
                    loss_mbce_extra_bg = 0

                loss_mbce_fake = self.BCELoss_fake(
                    logits_for_fake[:, -self.n_new_classes:],
                    fake_label
                ).mean(dim=[0, 2, 3])

                stride_num = features[-1].shape[0] * features[-1].shape[2] * features[-1].shape[3]
                weight_extra_bg = self.extra_bg_ratio * region_bg.sum() / stride_num
                weight_fake = fake_label.shape[1] / stride_num

                loss_mbce = loss_mbce_ori + loss_mbce_fake * weight_fake + loss_mbce_extra_bg * weight_extra_bg
                loss_mbce = loss_mbce / (1 + weight_extra_bg + weight_fake)

                loss_pkd = self.PKDLoss(features, features_old, pseudo_label_region.to(torch.float32))

                loss_cont = self.ContLoss(
                    features[-1], logit[:, -self.n_new_classes:], data['label'], self.prev_prototypes)

                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() \
                       + self.config['hyperparameter']['pkd'] * loss_pkd.sum() \
                       + self.config['hyperparameter']['cont'] * loss_cont

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item() * self.config['hyperparameter']['mbce'])
            self.train_metrics.update('loss_pkd', loss_pkd.sum().item() * self.config['hyperparameter']['pkd'])
            self.train_metrics.update('loss_cont', loss_cont.item() * self.config['hyperparameter']['cont'])

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def compute_pred_number(self):
        self.logger.info("computing pred number of pixels...")

        pred_numbers = torch.zeros(self.n_old_classes + 1).to(self.device)

        for batch_idx, data in enumerate(self.train_loader):
            with torch.no_grad():
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                logit_old, _, _ = self.model_old(data['image'], ret_intermediate=False)

                logit_old = logit_old.detach()
                pred = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)
                pred_region = (pred * (data['label'] == 0))[:, 8::16, 8::16]

                real_bg_region = torch.logical_and(pred == 0, data['label'] == 0)[:, 8::16, 8::16]
                pred_numbers[0] = pred_numbers[0] + real_bg_region.sum()

                for i in range(1, self.n_old_classes + 1):
                    pred_numbers[i] = pred_numbers[i] + (pred_region == i).sum()

            self.progress(self.logger, batch_idx, len(self.train_loader))

        return pred_numbers
        # return 0
