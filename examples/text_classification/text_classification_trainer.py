#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import logging
import math
import os

import numpy as np
import sklearn
import torch
import wandb
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from torch.nn import CrossEntropyLoss

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


class TextClassificationTrainer:
    def __init__(self, args, tc_data_manager, pipe_transformer):
        self.args = args
        self.num_labels = args.num_labels

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # Pipe Transformer
        self.pipe_transformer = pipe_transformer
        self.frozen_model = None
        self.pipe_model = None
        self.train_dl = None
        self.test_dl = None
        self.device_first = None
        self.device_last = None

        self.tc_data_manager = tc_data_manager
        _, _, self.test_examples = tc_data_manager.get_dataset()

    def train_model(self):
        epoch_start = self.pipe_transformer.start()
        self.frozen_model, self.pipe_model, \
        self.train_dl, self.test_dl, \
        self.device_first, self.device_last = self.pipe_transformer.get_new_model_and_dataset()

        # start the training loop
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        for epoch in range(epoch_start, self.args.num_train_epochs):
            self.pipe_transformer.transform(epoch)
            self.frozen_model, self.pipe_model, \
            self.train_dl, self.test_dl, \
            self.device_first, self.device_last = self.pipe_transformer.get_new_model_and_dataset()

            iteration_in_total = len(
                self.train_dl) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

            optimizer, scheduler = self.build_optimizer(self.pipe_model, iteration_in_total)

            self.pipe_model.train()
            if self.frozen_model is not None:
                self.frozen_model.eval()

            for batch_idx, batch in enumerate(self.train_dl):
                if batch_idx == 0:
                    self.pipe_model._sync_params()

                batch = tuple(t for t in batch)
                # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                sample_index_list = batch[0].to(self.device_first).cpu().numpy()
                x = batch[1].to(self.device_first)
                labels = batch[4].to(self.device_last)

                # logging.info(batch)
                # logging.info(sample_index_list)

                logits = self.pipe_transformer.forward(epoch, batch_idx, sample_index_list, x, True, True)
                # logits = self.pipe_model(x)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                # logging.info(loss)
                current_loss = loss.item()
                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), current_loss))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.pipe_model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.pipe_model.zero_grad()
                    global_step += 1

                    if self.args.evaluate_during_training and (self.args.evaluate_during_training_steps > 0
                                                               and global_step % self.args.evaluate_during_training_steps == 0):
                        results, _, _ = self.eval_model(epoch, global_step)
                        logging.info(results)

                if self.args.is_debug_mode == 1 and global_step > 3:
                    break
        results, _, _ = self.eval_model(self.args.num_train_epochs-1, global_step)
        logging.info(results)
        return global_step, tr_loss / global_step

    def eval_model(self, epoch, global_step):
        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = self.tc_data_manager.get_test_sample_len(epoch)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        self.pipe_model.eval()
        logging.info("len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches))
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t for t in batch)

                sample_index_list = batch[0].to(self.device_first).cpu().numpy()
                if i == len(self.test_dl) - 1:
                    logging.info(batch)
                x = batch[1].to(self.device_first)
                labels = batch[4].to(self.device_last)

                logits = self.pipe_transformer.forward(epoch, i, sample_index_list, x, False, False)
                if i == len(self.test_dl) - 1:
                    logging.info("i = " + str(i))
                    logging.info("sample_index_list = " + str(sample_index_list))
                    logging.info("x = %s, x.len = %d" % (str(x), len(x)))
                    logging.info(labels)
                    logging.info(logits)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()
                # logging.info("test. batch index = %d, loss = %s" % (i, str(eval_loss)))

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        # logging.info("preds = " + str(preds))
        # logging.info("out_label_ids = " + str(out_label_ids))
        result, wrong = self.compute_metrics(preds, out_label_ids, self.test_examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        os.makedirs(self.args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        if result["acc"] > self.best_accuracy:
            self.best_accuracy = result["acc"]
        logging.info("best_accuracy = %f" % self.best_accuracy)
        if self.args.global_rank == 0:
            wandb.log({"Evaluation Accuracy (best)": self.best_accuracy, "step": global_step})
            wandb.log({"Evaluation Accuracy": result["acc"], "step": global_step})
            wandb.log({"Evaluation Loss": result["eval_loss"], "step": global_step})

        self.results.update(result)
        logging.info(self.results)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % self.args.warmup_steps)
        # optimizer = torch.optim.Adam(self._get_optimizer_grouped_parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler
