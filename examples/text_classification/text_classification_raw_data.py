"""
An example of running centralized experiments of fed-transformer models in FedNLP.
Example usage: 
(under the root folder)
python -m experiments.centralized.transformer_exps.text_classification_raw_data \
    --dataset sentiment_140 \
    --data_file data/data_loaders/sentiment_140_data_loader.pkl \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/sentiment_140_fed/ \
    --fp16
"""
import argparse
import logging
import os
import sys

import pandas as pd
import sklearn
# this is a temporal import, we will refactor FedML as a package installation
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from model.nlp.classification.classification_model import ClassificationModel
import pipe_transformer.data.SST_2.data_loader


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Data related
    parser.add_argument('--dataset', type=str, default='20news', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../data/text_classification/20Newsgroups/20news-18828',
                        help='data directory')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/20news_data_loader.pkl',
                        help='data pickle file')

    # Model related
    parser.add_argument('--model_type', type=str, default='distilbert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    # Learning related

    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--num_train_epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')
    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    args = parser.parse_args()

    return args


def load_data(args, dataset):
    print("Loading dataset = %s" % dataset)
    # all_data = pickle.load(open(args.data_file, "rb"))
    data_loader = pipe_transformer.data.SST_2.data_loader.RawDataLoader(args.data_dir)
    all_data = data_loader.data_loader()

    X, Y, target_vocab, attributes = all_data["X"], all_data["Y"], all_data["target_vocab"], all_data["attributes"]
    train_data = [(X[idx], target_vocab[Y[idx]]) for idx in attributes["train_index_list"]]
    test_data = [(X[idx], target_vocab[Y[idx]]) for idx in attributes["test_index_list"]]
    return pd.DataFrame(train_data), pd.DataFrame(test_data), len(target_vocab)


def main(args):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Loading full data (for centralized learning)
    train_data, test_data, num_labels = load_data(args, args.dataset)

    # Create a ClassificationModel.
    model = ClassificationModel(
        args.model_type, args.model_name, num_labels=num_labels,
        args={"num_train_epochs": args.num_train_epochs,
              "learning_rate": args.learning_rate,
              "gradient_accumulation_steps": args.gradient_accumulation_steps,
              "do_lower_case": args.do_lower_case,
              "manual_seed": args.manual_seed,
              "reprocess_input_data": True,
              "overwrite_output_dir": True,
              "max_seq_length": args.max_seq_length,
              "train_batch_size": args.train_batch_size,
              "eval_batch_size": args.eval_batch_size,
              "fp16": args.fp16,
              "n_gpu": args.n_gpu,
              "output_dir": args.output_dir})

    # Strat training.
    model.train_model(train_data)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(
        test_data, acc=sklearn.metrics.accuracy_score)
    logging.info("eval_res=%s" % (str(result)))


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    run = wandb.init(project="pipe_and_ddp",
                     name="PipeTransformer""-" + str(args.dataset),
                     config=args)
    main(args)
