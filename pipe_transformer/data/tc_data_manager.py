import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DistributedSampler, DataLoader, TensorDataset, RandomSampler

from .base_data_manager import BaseDataManager
from ..data.SST_2.classification_utils import convert_examples_to_features
from ..data.SST_2.data_loader import RawDataLoader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]


class TCDatasetManager(BaseDataManager):
    def __init__(self, model_args, args, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.tokenizer = tokenizer

        self.num_labels = -1
        self.train_dataset, self.test_dataset, self.test_examples = self.load_data(
            data_dir=model_args.data_dir, dataset=model_args.dataset)

        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None
        self.train_sample_idx_list_by_epoch = dict()
        self.test_sample_idx_list_by_epoch = dict()
        self.latest_train_sample_idx_list = dict()
        self.latest_test_sample_idx_list = dict()

    def load_data(self, data_dir, dataset):
        print("Loading dataset = %s" % dataset)
        # all_data = pickle.load(open(args.data_file, "rb"))
        data_loader = RawDataLoader(data_dir)
        all_data = data_loader.data_loader()

        X, Y, target_vocab, attributes = all_data["X"], all_data["Y"], all_data["target_vocab"], all_data["attributes"]
        train_data = [(X[idx], target_vocab[Y[idx]], idx) for idx in attributes["train_index_list"]]
        test_data = [(X[idx], target_vocab[Y[idx]], idx) for idx in attributes["test_index_list"]]
        self.num_labels = len(target_vocab)

        # training dataset
        train_df = pd.DataFrame(train_data)
        train_examples = [
            InputExample(oid, text, None, label)
            for i, (text, label, oid) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1], train_df.iloc[:, 2]))
        ]
        train_dataset = self.load_and_cache_examples(train_examples)

        # test dataset
        test_df = pd.DataFrame(test_data)
        eval_examples = [
            InputExample(i, text, None, label)
            for i, (text, label, oid) in enumerate(zip(test_df.iloc[:, 0], test_df.iloc[:, 1], train_df.iloc[:, 2]))
        ]
        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True
        )

        return train_dataset, eval_dataset, eval_examples

    def get_dataset(self):
        return self.train_dataset, self.test_dataset, self.test_examples

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, silent=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        process_count = self.model_args.process_count

        tokenizer = self.tokenizer
        args = self.model_args

        if not no_cache:
            no_cache = args.no_cache

        output_mode = "classification"

        if not no_cache:
            os.makedirs(args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
            ),
        )
        logging.info("cached_features_file = %s" % str(cached_features_file))
        logging.info("args.reprocess_input_data = %s" % str(args.reprocess_input_data))
        logging.info("no_cache = %s" % str(no_cache))
        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            logging.info(f" Features loaded from cache at {cached_features_file}")
        else:
            logging.info(" Converting to features started. Cache is not used.")

            # If labels_map is defined, then labels need to be replaced with ints
            if args.labels_map and not args.regression:
                for example in examples:
                    example.label = args.labels_map[example.label]

            features = convert_examples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=False,
                silent=args.silent or silent,
                use_multiprocessing=args.use_multiprocessing,
                sliding_window=args.sliding_window,
                flatten=not evaluate,
                stride=args.stride,
                add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # avoid padding in case of single example/online inferencing to decrease execution time
                pad_to_max_length=bool(len(examples) > 1),
                args=args,
            )
            logging.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)

        all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    def get_data_loader_with_node_rank(self, epoch, batch_size, node_rank, num_replicas, local_rank):
        logging.info("---node_rank = %d, num_replicas = %d, local_rank = %d --------------" % (
            node_rank, num_replicas, local_rank))
        logging.info("train dataset len = %d, test dataset len = %d" % (len(self.train_dataset), len(self.test_dataset)))

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
