import argparse
from argparse import REMAINDER
import yaml
import logging


def add_args():
    parser = argparse.ArgumentParser(
        description="PipeTransformer: "
                    "Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")
    parser.add_argument(
        "--yaml_config_file",
        help="yaml configuration file",
        type=str,
        required=True,
    )

    # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--master_addr", type=str, default="192.168.11.2")

    parser.add_argument("--master_port", type=int, default=22222)

    parser.add_argument("--if_name", type=str, default="lo")

    parser.add_argument("--is_distributed", default=1, type=int,
                        help="is_distributed")

    parser.add_argument("--pipe_len_at_the_beginning", default=4, type=int,
                        help="pipe_len_at_the_beginning")

    parser.add_argument("--is_infiniband", default=1, type=int,
                        help="is_infiniband")

    parser.add_argument("--num_chunks_of_micro_batches", default=1 * 8, type=int,
                        help="num_chunks_of_micro_batches")

    parser.add_argument("--freeze_strategy_alpha", type=float, default=0.5)

    parser.add_argument('--freeze', dest='b_freeze', action='store_true')
    parser.add_argument('--no_freeze', dest='b_freeze', action='store_false')
    parser.set_defaults(b_freeze=True)

    parser.add_argument('--auto_pipe', dest='b_auto_pipe', action='store_true')
    parser.add_argument('--no_auto_pipe', dest='b_auto_pipe', action='store_false')
    parser.set_defaults(b_auto_pipe=True)

    parser.add_argument('--auto_dp', dest='b_auto_dp', action='store_true')
    parser.add_argument('--no_auto_dp', dest='b_auto_dp', action='store_false')
    parser.set_defaults(b_auto_dp=True)

    parser.add_argument('--cache', dest='b_cache', action='store_true')
    parser.add_argument('--no_cache', dest='b_cache', action='store_false')
    parser.set_defaults(b_cache=True)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # model related
    parser.add_argument('--model', type=str, default='transformer', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar100', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./data/cifar100',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.3)

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument("--warmup_steps", default=30, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--epochs', type=int, default=300, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument("--freq_eval_train_acc", default=4, type=int)

    parser.add_argument("--freq_eval_test_acc", default=1, type=int)

    parser.add_argument("--pretrained_dir", type=str,
                        default="./../../model/cv/pretrained/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    args = parser.parse_args()
    return args


class Arguments:
    """Argument class which contains all arguments from yaml config and constructs additional arguments"""

    def __init__(self, cmd_args):
        # set the command line arguments
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

        self.yaml_paths = [cmd_args.yaml_config_file]
        # Load all arguments from yaml config
        # https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started
        configuration = self.load_yaml_config(cmd_args.yaml_config_file)

        # Override class attributes from current yaml config
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")


def get_arguments():
    cmd_args = add_args()

    # Load all arguments from YAML config file
    args = Arguments(cmd_args)

    logging.info("local_rank = %d" % args.local_rank)
    if args.local_rank == 0:
        logging.info(vars(args))
    return args
