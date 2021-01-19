import logging
import time

import torch
import wandb
from torch import nn

from examples.image_classification.utils import WarmupCosineSchedule, WarmupLinearSchedule


class CVTrainer:
    def __init__(self, args, pipe_transformer):
        self.args = args
        self.pipe_transformer = pipe_transformer

        self.frozen_model = None
        self.pipe_model = None
        self.train_dl = None
        self.test_dl = None
        self.device_first = None
        self.device_last = None

    def train_and_eval(self):
        epoch_start = self.pipe_transformer.start()
        for epoch in range(epoch_start, self.args.epochs):

            self.pipe_transformer.transform(epoch)

            self.frozen_model, self.pipe_model, \
            self.train_dl, self.test_dl, \
            self.device_first, self.device_last = self.pipe_transformer.get_new_model_and_dataset()

            self.train(epoch)
            self.eval(epoch)

    def train(self, epoch):

        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = self.build_optimizer(self.pipe_model)

        # measure latency with cuda event:
        # https://discuss.pytorch.org/t/distributed-training-slower-than-dataparallel/81539/4
        self.pipe_model.train()
        if self.frozen_model is not None:
            self.frozen_model.eval()

        iteration_num = 0
        num_sample_processed_in_total = 0
        communication_count = 0.0

        starting_time_forward = 0.0
        forward_time_accumulate = 0.0

        backwards_time_accumulate = 0.0
        for batch_idx, (sample_index_list, x, target) in enumerate(self.train_dl):
            communication_count += 1
            iteration_num += 1

            if batch_idx == 0:
                starting_time = time.time()
                self.pipe_model._sync_params()

            if batch_idx > 0:
                backwards_time_accumulate += time.time() - starting_time_forward
                backwards_time_per_batch = backwards_time_accumulate / (batch_idx+1)
                logging.critical("(epoch = %d) backwards_time_per_batch = %s" % (epoch, backwards_time_per_batch))

            logging.info("--------------global_rank = %d. Epoch %d, batch index %d Statistics: " % (
                self.args.global_rank, epoch, batch_idx))
            logging.info("global_rank = %d. epoch = %d, batch index = %d/%d" % (
                self.args.global_rank, epoch, batch_idx, len(self.train_dl) - 1))
            num_sample_processed_in_total += len(x)

            sample_index_list = sample_index_list.cpu().numpy()
            x = x.to(self.device_first)
            target = target.to(self.device_last)

            optimizer.zero_grad()

            starting_time_forward = time.time()
            log_probs = self.pipe_transformer.forward(epoch, batch_idx, sample_index_list, x, True, True)

            end_time_forward = time.time()
            forward_time_accumulate += (end_time_forward - starting_time_forward)
            forward_time_per_batch = forward_time_accumulate / (batch_idx + 1)
            logging.critical("(epoch = %d) forward_time_per_batch = %s" % (epoch, forward_time_per_batch))

            loss = criterion(log_probs, target)
            loss.backward()
            # this clip will cost 0.6 second, can be skipped?
            torch.nn.utils.clip_grad_norm_(self.pipe_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            self.pipe_transformer.collect_freeze_info()

            if batch_idx == 0:
                time_finish_prepare_ddp = time.time()
                logging.info("global_rank = %d. data loading cost = %s" % (
                    self.args.global_rank, str(time_finish_prepare_ddp - starting_time)))

            sample_num_throughput = int(num_sample_processed_in_total / (time.time() - time_finish_prepare_ddp)) * self.pipe_transformer.get_active_world_size()
            logging.critical("global_rank = %d. sample_num_throughput (images/second): %d" % (self.args.local_rank, sample_num_throughput))

            comm_freq = communication_count / (time.time() - time_finish_prepare_ddp)
            logging.critical("global_rank = %d. communication frequency (cross machine sync/second): %f" % (self.args.global_rank, comm_freq))

            if batch_idx == len(self.train_dl) - 1 and self.args.global_rank == 0:
                wandb.log({"comm_frequency": comm_freq, "epoch": epoch})
                wandb.log({"sample_throughput": sample_num_throughput, "epoch": epoch})

                forward_time_per_batch = forward_time_accumulate / len(self.train_dl)
                logging.critical("(epoch = %d) forward_time_per_batch = %s" % (epoch, forward_time_per_batch))
                wandb.log({"forward_time_per_batch": forward_time_per_batch, "epoch": epoch})

            logging.info("-------------------------------------")
            if iteration_num == 3 and self.args.is_debug_mode:
                break
        if self.args.global_rank == 0:
            backwards_time_per_batch = backwards_time_accumulate / len(self.train_dl)
            wandb.log({"backwards_time_per_batch": backwards_time_per_batch, "epoch": epoch})
            logging.critical("(epoch = %d) backwards_time_per_batch = %s" % (epoch, backwards_time_per_batch))

    def eval(self, epoch):
        # train data
        # if epoch == self.args.epochs - 1:
        if (epoch + 1) % self.args.freq_eval_train_acc == 0 or epoch == self.args.epochs - 1:

            train_tot_correct, train_num_sample, train_loss = self._infer(self.train_dl, epoch, is_train=True)
            # test on training dataset
            train_acc = train_tot_correct / train_num_sample
            train_loss = train_loss / train_num_sample

            if self.args.global_rank == 0:
                wandb.log({"Train/Acc": train_acc, "epoch": epoch})
                wandb.log({"Train/Loss": train_loss, "epoch": epoch})
                stats = {'training_acc': train_acc, 'training_loss': train_loss}
                logging.critical(stats)

        # test data
        # if epoch == self.args.epochs - 1:
        if (epoch + 1) % self.args.freq_eval_test_acc == 0 or epoch == self.args.epochs - 1:

            test_tot_correct, test_num_sample, test_loss = self._infer(self.test_dl, epoch, is_train=False)

            # test on test dataset
            test_acc = test_tot_correct / test_num_sample
            test_loss = test_loss / test_num_sample

            if self.args.global_rank == 0:
                wandb.log({"Test/Acc": test_acc, "epoch": epoch})
                wandb.log({"Test/Loss": test_loss, "epoch": epoch})
                stats = {'test_acc': test_acc, 'test_loss': test_loss}
                logging.critical(stats)

    def _infer(self, test_data, epoch, is_train):
        if self.frozen_model is not None:
            self.frozen_model.eval()
        self.pipe_model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss()
        iteration_num = 0

        starting_time_forward = 0.0
        forward_time_accumulate = 0.0

        backwards_time_accumulate = 0.0
        with torch.no_grad():
            for batch_idx, (sample_index_list, x, target) in enumerate(test_data):
                logging.info("(%s)evaluation - batch index = %d/%d" % (str(is_train), batch_idx, len(test_data) - 1))

                if batch_idx > 0:
                    backwards_time_accumulate += time.time() - starting_time_forward
                    backwards_time_per_batch = backwards_time_accumulate / (batch_idx + 1)
                    logging.critical("(epoch = %d) backwards_time_per_batch = %s" % (epoch, backwards_time_per_batch))

                iteration_num += 1
                sample_index_list = sample_index_list.cpu().numpy()
                x = x.to(self.device_first)
                target = target.to(self.device_last)

                starting_time_forward = time.time()
                if is_train:
                    log_probs = self.pipe_transformer.forward(epoch, batch_idx, sample_index_list, x, False, True)
                else:
                    log_probs = self.pipe_transformer.forward(epoch, batch_idx, sample_index_list, x, False, False)
                end_time_forward = time.time()
                forward_time_accumulate += (end_time_forward - starting_time_forward)
                forward_time_per_batch = forward_time_accumulate / (batch_idx + 1)
                logging.critical("(epoch = %d) forward_time_per_batch = %s" % (epoch, forward_time_per_batch))

                loss = criterion(log_probs, target)
                _, predicted = torch.max(log_probs, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
                if iteration_num == 3 and self.args.is_debug_mode:
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