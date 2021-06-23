import logging
import os
import socket
import sys

import psutil
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from pipe_transformer.config_args import ConfigArgs
from pipe_transformer.pipe_transformer import PipeTransformer

from pipe_transformer.data.cv_data_manager import CVDatasetManager
from model.cv.vision_transformer_origin import CONFIGS
from model.cv.vision_transformer_origin import VisionTransformer

from examples.image_classification.cv_trainer import CVTrainer
from examples.image_classification.arguments import get_arguments


def post_complete_message_to_sweep(args, config):
    pipe_path = "/tmp/pipe_transformer_training_status_cv"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n%s" % (str(args), str(config)))


if __name__ == "__main__":
    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    args = get_arguments()

    hostname = socket.gethostname()
    logging.info("#############process ID = " +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    logging.info(args)
    """
        Dataset related
    """
    cv_data_manager = CVDatasetManager(args)
    cv_data_manager.set_seed(0)
    output_dim = cv_data_manager.get_output_dim()

    """
        Model related
    """
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    model_config = CONFIGS[model_type]
    model_config.output_dim = output_dim
    args.num_layer = model_config.transformer.num_layers
    args.transformer_hidden_size = model_config.hidden_size
    args.seq_len = 197

    logging.info("Vision Transformer Configuration: " + str(model_config))
    model = VisionTransformer(model_config, args.img_size, zero_head=True, num_classes=output_dim, vis=False)
    # model.load_from(np.load(args.pretrained_dir))
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info("model_size = " + str(model_size))

    num_layers = model_config.transformer.num_layers
    logging.info("num_layers = %d" % num_layers)

    """
        PipeTransformer related
    """
    config = ConfigArgs()
    config.b_auto_dp = args.b_auto_dp
    config.b_freeze = args.b_freeze
    config.b_auto_pipe = args.b_auto_pipe
    config.b_cache = args.b_cache
    config.freeze_strategy_alpha = args.freeze_strategy_alpha

    config.is_infiniband = args.is_infiniband
    config.master_addr = args.master_addr
    config.master_port = args.master_port
    config.if_name = args.if_name
    config.num_nodes = args.nnodes
    config.node_rank = args.node_rank
    config.local_rank = args.local_rank

    config.pipe_len_at_the_beginning = args.pipe_len_at_the_beginning
    config.num_chunks_of_micro_batches = args.num_chunks_of_micro_batches

    config.learning_task = config.LEARNING_TASK_IMAGE_CLASSIFICATION
    config.model_name = config.MODEL_VIT
    config.num_layer = num_layers
    config.output_dim = output_dim
    config.hidden_size = args.transformer_hidden_size
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.epochs = args.epochs

    config.is_debug_mode = args.is_debug_mode

    logging.info(config)

    pipe_transformer = PipeTransformer(config, cv_data_manager, model_config, model)
    args.global_rank = pipe_transformer.get_global_rank()
    logging.info("transformer is initialized")
    logging.info(args)
    """
        Logging related
    """
    if args.global_rank == 0:
        run = wandb.init(project="pipe_and_ddp",
                         name="PipeTransformer""-r" + str(args.run_id) + "-" + str(args.dataset),
                         config=args)

    """
        Trainer related
    """
    trainer = CVTrainer(args, pipe_transformer)
    trainer.train_and_eval()

    """
        PipeTransformer related
    """
    pipe_transformer.finish()

    if args.global_rank == 0:
        wandb.finish()

    if args.local_rank == 0:
        post_complete_message_to_sweep(args, config)
