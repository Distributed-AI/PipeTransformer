import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DistributedSampler

from examples.question_answering.question_answering_utils import (
    get_examples,
    squad_convert_examples_to_features,
)
from .SQuAD_1_1.data_loader import RawDataLoader
from .base_data_manager import BaseDataManager


class QADatasetManager(BaseDataManager):
    def __init__(self, model_args, args, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.tokenizer = tokenizer

        self.train_dataset, self.test_dataset, \
        self.train_data, self.test_data, \
        self.examples, self.features = self.load_data(data_dir=model_args.data_dir, dataset=model_args.dataset)

        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None
        self.train_sample_idx_list_by_epoch = dict()
        self.test_sample_idx_list_by_epoch = dict()
        self.latest_train_sample_idx_list = dict()
        self.latest_test_sample_idx_list = dict()

    def get_dataset(self):
        return self.train_dataset, self.test_dataset, \
        self.train_data, self.test_data, \
        self.examples, self.features

    def load_data(self, data_dir, dataset):
        print("Loading dataset = %s" % dataset)
        assert dataset in ["squad_1.1"]
        # all_data = pickle.load(open(args.data_file, "rb"))
        data_loader = RawDataLoader(data_dir)
        all_data = data_loader.data_loader()

        context_X, question_X, question_ids, Y, attributes = all_data["context_X"], all_data["question_X"], \
                                                             all_data["question_ids"], all_data["Y"], all_data[
                                                                 "attributes"]

        def get_data_by_index_list(dataset, index_list):
            data = dict()
            for key in dataset.keys():
                data[key] = []
            for idx in index_list:
                for key in dataset.keys():
                    data[key].append(dataset[key][idx])
            data["original_index"] = index_list
            return data

        input_dataset = {"context_X": context_X, "question_X": question_X, "question_ids": question_ids, "Y": Y}
        train_data = get_data_by_index_list(input_dataset, attributes["train_index_list"])
        test_data = get_data_by_index_list(input_dataset, attributes["test_index_list"])

        train_data, train_id_mapping_dict = self._get_normal_format(train_data, cut_off=None)
        test_data, test_id_mapping_dict = self._get_normal_format(test_data, cut_off=None)

        if isinstance(train_data, str):
            with open(train_data, "r", encoding=self.args.encoding) as f:
                train_examples = json.load(f)
        else:
            train_examples = train_data

        train_dataset = self.load_and_cache_examples(train_examples)

        if isinstance(test_data, str):
            with open(test_data, "r", encoding=self.args.encoding) as f:
                eval_examples = json.load(f)
        else:
            eval_examples = test_data

        eval_dataset, examples, features = self.load_and_cache_examples(
            eval_examples, evaluate=True, output_examples=True
        )
        return train_dataset, eval_dataset, train_data, test_data, examples, features

    def _get_normal_format(self, dataset, cut_off=None):
        """
        reformat the dataset to normal version.
        """
        reformatted_data = []
        id_mapping_dict = dict()
        assert len(dataset["context_X"]) == len(dataset["question_X"]) == len(dataset["Y"]) == len(
            dataset["question_ids"]) == len(dataset["original_index"])
        for c, q, a, qid, oid in zip(dataset["context_X"], dataset["question_X"], dataset["Y"],
                                     dataset["question_ids"],
                                     dataset["original_index"]):
            item = {}
            item["context"] = c
            id_mapping_dict[len(reformatted_data) + 1] = oid
            item["qas"] = [
                {
                    "oid": oid,
                    "id": "%d" % (len(reformatted_data) + 1),
                    "qid": qid,
                    "is_impossible": False,
                    "question": q,
                    "answers": [{"text": c[a[0]:a[1]], "answer_start": a[0]}],
                }
            ]
            reformatted_data.append(item)
        return reformatted_data[:cut_off], id_mapping_dict

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False):
        """
        Converts a list of examples to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        tokenizer = self.tokenizer
        args = self.model_args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(args.cache_dir, exist_ok=True)

        examples = get_examples(examples, is_training=not evaluate)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir, "cached_{}_{}_{}_{}".format(mode, args.model_type, args.max_seq_length, len(examples)),
        )
        logging.info("cached_features_file = %s" % cached_features_file)
        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features)
        ):
            features = torch.load(cached_features_file)
            logging.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logging.info(" Converting to features started.")

            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                tqdm_enabled=not args.silent,
                threads=args.process_count,
                args=args,
            )
            if not no_cache:
                torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_original_id = torch.tensor([f.original_id for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if evaluate:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_original_id, all_input_ids, all_attention_masks,
                all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_original_id,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        if output_examples:
            return dataset, examples, features
        return dataset

    def get_data_loader_with_node_rank(self, epoch, batch_size, node_rank, num_replicas, local_rank):
        logging.info("---node_rank = %d, num_replicas = %d, local_rank = %d --------------" % (
            node_rank, num_replicas, local_rank))
        logging.info(
            "train dataset len = %d, test dataset len = %d" % (len(self.train_dataset), len(self.test_dataset)))

        if self.train_sampler is not None:
            del self.train_sampler
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=num_replicas, rank=local_rank)
        indexes = list(iter(self.train_sampler))
        logging.info("global_rank = %d. train indexes len = %d" % (self.args.global_rank, len(indexes)))
        self.train_sample_idx_list_by_epoch[epoch] = indexes
        self.latest_train_sample_idx_list = indexes

        if self.train_loader is not None:
            del self.train_loader
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=self.train_sampler,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

        # TEST
        if self.test_sampler is not None:
            del self.test_sampler
        self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=num_replicas,
                                               rank=local_rank, drop_last=False)
        indexes = list(iter(self.test_sampler))
        logging.info("global_rank = %d. test indexes len = %d" % (self.args.global_rank, len(indexes)))
        self.test_sample_idx_list_by_epoch[epoch] = indexes
        self.latest_test_sample_idx_list = indexes

        if self.test_loader is not None:
            del self.test_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      sampler=self.test_sampler,
                                      batch_size=self.train_batch_size,
                                      num_workers=0,
                                      pin_memory=True,
                                      drop_last=False)
        return self.train_loader, self.test_loader

    def get_train_sample_index(self, epoch):
        return self.train_sample_idx_list_by_epoch[epoch]

    def get_test_sample_index(self, epoch):
        return self.test_sample_idx_list_by_epoch[epoch]

    def get_train_sample_len(self, epoch):
        if epoch not in self.train_sample_idx_list_by_epoch.keys():
            return len(self.latest_train_sample_idx_list)
        return len(self.train_sample_idx_list_by_epoch[epoch])

    def get_test_sample_len(self, epoch):
        if epoch not in self.test_sample_idx_list_by_epoch.keys():
            return len(self.latest_test_sample_idx_list)
        return len(self.test_sample_idx_list_by_epoch[epoch])

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
