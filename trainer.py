import logging
import time

import torch
import wandb
from torch import nn

from utils import WarmupCosineSchedule, WarmupLinearSchedule


class VisionTransformerTrainer:
    def __init__(self, args, auto_freeze, auto_pipe, auto_dp, auto_cache,
                 frozen_model, pipe_model,
                 cv_data_manager):
        self.args = args

        # auto
        self.auto_freeze = auto_freeze
        self.auto_pipe = auto_pipe
        self.auto_dp = auto_dp
        self.auto_cache = auto_cache

        self.device_first = self.auto_pipe.get_device_first()
        self.device_last = self.auto_pipe.get_device_last()

        # model
        self.frozen_model = frozen_model
        self.pipe_model = pipe_model

        # data
        self.cv_data_manager = cv_data_manager
        self.train_dl, self.test_dl = cv_data_manager.get_data_loader(args.batch_size, auto_dp.get_data_duplicate_num(),
                                                            auto_dp.get_data_rank())

    def train_and_eval(self, freeze_point):
        epoch_start = freeze_point['epoch']
        for epoch in range(epoch_start, self.args.epochs):
            self.train(epoch)
            self.eval(epoch)

    def train(self, epoch):
        if self.auto_freeze.is_freeze_open():
            new_freeze_point = dict()
            new_freeze_point['epoch'] = epoch

            # self, auto_pipe, frozen_model, pipe_model, num_frozen_layers, freeze_point
            self.frozen_model, self.pipe_model, \
            is_pipe_len_changed, is_frozen_layer_changed = self.auto_dp.transform(self.auto_pipe,
                                                                                  self.frozen_model,
                                                                                  self.pipe_model,
                                                                                  self.auto_freeze.get_hand_crafted_frozen_layers_by_epoch(epoch),
                                                                                  new_freeze_point)
            new_freeze_point = self.auto_dp.get_freeze_point()
            if is_pipe_len_changed:
                self.train_dl, self.test_dl = self.cv_data_manager.get_data_loader(self.args.batch_size,
                                                                                 self.auto_dp.get_data_duplicate_num(),
                                                                                 self.auto_dp.get_data_rank())
                self.device_first = self.auto_pipe.get_device_first()
                self.device_last = self.auto_pipe.get_device_last()

            logging.info("global_rank = %d. is_frozen_layer_changed: %s" % (self.auto_dp.get_global_rank(), str(is_frozen_layer_changed)))
            if is_frozen_layer_changed:
                self.auto_cache.update_num_frozen_layers(self.auto_pipe.get_num_frozen_layers(),
                                                         len(self.train_dl),
                                                         len(self.test_dl))

        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = self.build_optimizer(self.pipe_model)

        logging.info("device_first = " + str(self.device_first))
        logging.info("device_last = " + str(self.device_last))

        # measure latency with cuda event:
        # https://discuss.pytorch.org/t/distributed-training-slower-than-dataparallel/81539/4
        self.pipe_model.train()
        if self.frozen_model is not None:
            self.frozen_model.eval()

        iteration_num = 0
        num_sample_processed_in_total = 0
        communication_count = 0.0
        logging.info("global_rank = %d. data_loader id = %d/" % (self.auto_dp.get_global_rank(), id(self.train_dl)))
        for batch_idx, (x, target) in enumerate(self.train_dl):
            # torch.cuda.empty_cache()

            if batch_idx == 0:
                starting_time = time.time()
            logging.info("--------------global_rank = %d. Epoch %d, batch index %d Statistics: " % (
                self.auto_dp.get_global_rank(), epoch, batch_idx))
            logging.info("global_rank = %d. epoch = %d, batch index = %d/%d" % (
                self.auto_dp.get_global_rank(), epoch, batch_idx, len(self.train_dl)))
            num_sample_processed_in_total += len(x)
            communication_count += 1
            iteration_num += 1

            x = x.to(self.device_first)
            target = target.to(self.device_last)

            optimizer.zero_grad()
            log_probs = self.auto_cache.infer_train(self.frozen_model, self.pipe_model, x, batch_idx)

            loss = criterion(log_probs, target)
            loss.backward()
            # this clip will cost 0.6 second, can be skipped?
            torch.nn.utils.clip_grad_norm_(self.pipe_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx == 0:
                time_finish_prepare_ddp = time.time()
                logging.info("global_rank = %d. data loading cost = %s" % (
                    self.auto_dp.get_global_rank(), str(time_finish_prepare_ddp - starting_time)))

            sample_num_throughput = int(
                num_sample_processed_in_total / (
                        time.time() - time_finish_prepare_ddp)) * self.auto_dp.get_active_world_size()
            logging.info("global_rank = %d. sample_num_throughput (images/second): %d" % (self.auto_dp.get_global_rank(),
                                                                                          sample_num_throughput))

            comm_freq = communication_count / (time.time() - time_finish_prepare_ddp)
            logging.info(
                "global_rank = %d. communication frequency (cross machine sync/second): %f" % (
                    self.auto_dp.get_global_rank(),
                    comm_freq))

            if batch_idx == len(self.train_dl) - 1 and self.auto_dp.get_global_rank() == 0:
                wandb.log({"comm_frequency": comm_freq, "epoch": epoch})
                wandb.log({"sample_throughput": sample_num_throughput, "epoch": epoch})
            logging.info("-------------------------------------")
            if iteration_num == 2 and self.args.is_debug_mode:
                break

    def eval(self, epoch):
        # train data
        # if epoch == self.args.epochs - 1:
        if (epoch + 1) % self.args.freq_eval_train_acc == 0 or epoch == self.args.epochs - 1:

            train_tot_correct, train_num_sample, train_loss = self._infer(self.train_dl, is_train=True)
            # test on training dataset
            train_acc = train_tot_correct / train_num_sample
            train_loss = train_loss / train_num_sample

            if self.args.global_rank == 0:
                wandb.log({"Train/Acc": train_acc, "epoch": epoch})
                wandb.log({"Train/Loss": train_loss, "epoch": epoch})
                stats = {'training_acc': train_acc, 'training_loss': train_loss}
                logging.info(stats)

        # test data
        # if epoch == self.args.epochs - 1:
        if (epoch + 1) % self.args.freq_eval_test_acc == 0 or epoch == self.args.epochs - 1:

            test_tot_correct, test_num_sample, test_loss = self._infer(self.test_dl, is_train=False)

            # test on test dataset
            test_acc = test_tot_correct / test_num_sample
            test_loss = test_loss / test_num_sample

            if self.args.global_rank == 0:
                wandb.log({"Test/Acc": test_acc, "epoch": epoch})
                wandb.log({"Test/Loss": test_loss, "epoch": epoch})
                stats = {'test_acc': test_acc, 'test_loss': test_loss}
                logging.info(stats)


    def _infer(self, test_data, is_train):
        if self.frozen_model is not None:
            self.frozen_model.eval()
        self.pipe_model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss()
        iteration_num = 0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                logging.info("evaluation - batch index = %d/%d" % (batch_idx, len(test_data)))
                iteration_num += 1
                x = x.to(self.device_first)
                target = target.to(self.device_last)

                if is_train:
                    log_probs = self.auto_cache.infer_train(self.frozen_model, self.pipe_model, x, batch_idx)
                else:
                    log_probs = self.auto_cache.infer_test(self.frozen_model, self.pipe_model, x, batch_idx)

                loss = criterion(log_probs, target)
                _, predicted = torch.max(log_probs, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
                if iteration_num == 2 and self.args.is_debug_mode:
                    break

        return test_acc, test_total, test_loss


    def build_optimizer(self, model):
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=self.args.lr,
                                        momentum=0.9,
                                        weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)

        if self.args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_steps,
                                             t_total=self.args.epochs)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps,
                                             t_total=self.args.epochs)
        return optimizer, scheduler



    def sync_all_devices(self, local_rank, device_cnt=4):
        for d in range(device_cnt):
            device = torch.device("cuda:" + str(local_rank + d))
            torch.cuda.synchronize(device)
