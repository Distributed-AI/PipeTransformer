from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os

import torch
from torch.nn import CrossEntropyLoss

from examples.question_answering.question_answering_utils import (
    RawResult,
    to_list,
    write_predictions,
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.testing_utils import _RunOutput

import wandb


class QuestionAnsweringTrainer:
    def __init__(self, args, qa_data_manager, pipe_transformer):
        """
        Initializes a QuestionAnsweringModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            args (optional): Default args will be used if this parameter is not provided. If provided,
                it should be a dict containing the args that should be changed in the default args'
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """  # noqa: ignore flake8"
        self.args = args

        # training results
        self.results = {}
        self.best_em = 0.0
        self.best_f1 = 0.0

        # Pipe Transformer
        self.pipe_transformer = pipe_transformer
        self.frozen_model = None
        self.pipe_model = None
        self.train_dl = None
        self.test_dl = None
        self.device_first = None
        self.device_last = None

        self.tc_data_manager = qa_data_manager
        self.train_dataset, self.test_dataset, \
               self.train_data, self.test_data, \
               self.examples, self.features = qa_data_manager.get_dataset()

    def train_model(self):
        epoch_start = self.pipe_transformer.start()
        self.frozen_model, self.pipe_model, \
        self.train_dl, self.test_dl, \
        self.device_first, self.device_last = self.pipe_transformer.get_new_model_and_dataset()

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
                batch = tuple(t for t in batch)
                # inputs = {"original_id": batch[0], "input_ids": batch[1], "attention_mask": batch[2],
                #           "token_type_ids": batch[3], "start_positions": batch[4], "end_positions": batch[5]}

                sample_index_list = batch[0].to(self.device_first).cpu().numpy()
                x = batch[1].to(self.device_first)
                start_positions = batch[4].to(self.device_last)
                end_positions = batch[5].to(self.device_last)

                logits = self.pipe_transformer.forward(epoch, batch_idx, sample_index_list, x, True, True)

                loss, _, _ = self._calculate_loss(logits, start_positions, end_positions)

                logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (epoch, batch_idx,
                                                                           len(self.train_dl), loss))
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
                        # results, _ = self.eval_model(eval_data, **kwargs)
                        result = self.eval_model_by_offical_script(epoch, global_step)
                        if result is not None:
                            logging.info("epoch = %d, global_step = %d, result = %s" % (epoch, global_step, str(result)))
                if global_step > 3 and self.args.is_debug_mode:
                    break
        result = self.eval_model_by_offical_script(self.args.num_train_epochs-1, global_step)
        if result is not None:
            logging.info("epoch = %d, global_step = %d, result = %s" % (self.args.num_train_epochs-1, global_step, str(result)))
        return global_step, tr_loss / global_step

    def _calculate_loss(self, logits, start_positions, end_positions):
        """
        From BertForQuestionAnswering
        """
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits

    def eval_model(self, eval_data, **kwargs):
        output_dir = self.args.output_dir

        all_predictions, all_nbest_json, scores_diff_json, eval_loss = self.evaluate(eval_data, output_dir)

        if isinstance(eval_data, str):
            with open(eval_data, "r", encoding=self.args.encoding) as f:
                truth = json.load(f)
        else:
            truth = eval_data

        result, texts = self.calculate_results(truth, all_predictions, **kwargs)
        result["eval_loss"] = eval_loss

        self.results.update(result)

        logging.info(self.results)

        return result, texts

    def eval_model_by_offical_script(self, epoch, step):

        all_predictions, all_nbest_json, scores_diff_json, eval_loss = self.evaluate(epoch)
        if self.args.global_rank != 0:
            return None

        prediction_dict = dict()
        for i, prediction in all_predictions.items():
            qid = self.test_data[int(i) - 1]["qas"][0]["qid"]
            prediction_dict[qid] = prediction

        with open(os.path.join(self.args.output_dir, "prediction.json"), "w") as f:
            json.dump(prediction_dict, f)

        f = os.popen("python ./evaluate-v1.1.py %s %s" % (
            self.args.eval_data_path, os.path.join(self.args.output_dir, "prediction.json")))

        result = eval(f.read().strip())
        em = float(result['exact_match'])
        f1 = float(result['f1'])
        if em > self.best_em:
            self.best_em = em
        if f1 > self.best_f1:
            self.best_f1 = f1

        logging.info("best_em = %f" % self.best_em)
        logging.info("best_f1 = %f" % self.best_f1)
        if self.args.global_rank == 0:
            wandb.log({"Evaluation EM (best)": self.best_em, "step": step})
            wandb.log({"Evaluation F1 (best)": self.best_f1, "step": step})
            wandb.log({"Evaluation EM": em, "step": step})
            wandb.log({"Evaluation F1": f1, "step": step})
        return result

    def evaluate(self, epoch):
        eval_loss = 0.0
        nb_eval_steps = 0

        all_results = []
        self.pipe_model.eval()
        for i, batch in enumerate(self.test_dl):
            logging.info("evaluation. epoch = %d, batch index = %d/%d" % (epoch, i, len(self.test_dl)-1))
            with torch.no_grad():
                batch = tuple(t for t in batch)
                # inputs = {"original_id": batch[0], "input_ids": batch[1], "attention_mask": batch[2],
                #           "token_type_ids": batch[3], "start_positions": batch[4], "end_positions": batch[5]}

                sample_index_list = batch[0].to(self.device_first).cpu().numpy()
                x = batch[1].to(self.device_first)
                start_positions = batch[4].to(self.device_last)
                end_positions = batch[5].to(self.device_last)
                example_indices = batch[4]

                logits = self.pipe_transformer.forward(epoch, i, sample_index_list, x, False, False)

                loss, start_logits, end_logits = self._calculate_loss(logits, start_positions, end_positions)
                eval_loss += loss

                for i, example_index in enumerate(example_indices):
                    eval_feature = self.features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    result = RawResult(
                        unique_id=unique_id,
                        start_logits=to_list(start_logits[i]),
                        end_logits=to_list(end_logits[i]),
                    )
                    all_results.append(result)

            nb_eval_steps += 1
            # if nb_eval_steps > 3 and self.args.is_debug_mode:
            #     break

        eval_loss = eval_loss / nb_eval_steps

        prefix = "test"
        os.makedirs(self.args.output_dir, exist_ok=True)

        output_prediction_file = os.path.join(self.args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.args.output_dir, "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file = os.path.join(self.args.output_dir, "null_odds_{}.json".format(prefix))

        """
        def write_predictions(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
):
        """
        all_predictions, all_nbest_json, scores_diff_json = write_predictions(
            self.examples,
            self.features,
            all_results,
            self.args.n_best_size,
            self.args.max_answer_length,
            False,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            False,
            True,
            self.args.null_score_diff_threshold,
        )

        return all_predictions, all_nbest_json, scores_diff_json, eval_loss

    def calculate_results(self, truth, predictions, **kwargs):
        truth_dict = {}
        questions_dict = {}
        for item in truth:
            for answer in item["qas"]:
                if answer["answers"]:
                    truth_dict[answer["id"]] = answer["answers"][0]["text"]
                else:
                    truth_dict[answer["id"]] = ""
                questions_dict[answer["id"]] = answer["question"]

        correct = 0
        incorrect = 0
        similar = 0
        correct_text = {}
        incorrect_text = {}
        similar_text = {}
        predicted_answers = []
        true_answers = []

        for q_id, answer in truth_dict.items():
            predicted_answers.append(predictions[q_id])
            true_answers.append(answer)
            if predictions[q_id].strip() == answer.strip():
                correct += 1
                correct_text[q_id] = answer
            elif predictions[q_id].strip() in answer.strip() or answer.strip() in predictions[q_id].strip():
                similar += 1
                similar_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }
            else:
                incorrect += 1
                incorrect_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                    "question": questions_dict[q_id],
                }

        extra_metrics = {}
        logging.info(kwargs)
        for metric, func in kwargs.items():
            logging.info("metric = %s, func = %s" % (str(metric), str(func)))
            extra_metrics[metric] = func(true_answers, predicted_answers)

        result = {"correct": correct, "similar": similar, "incorrect": incorrect, **extra_metrics}

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }

        return result, texts

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[1],
            "attention_mask": batch[2],
            "token_type_ids": batch[3],
            "start_positions": batch[4],
            "end_positions": batch[5],
        }

        return inputs

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
