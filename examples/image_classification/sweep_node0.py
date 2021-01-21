import argparse
import logging
import os
from time import sleep


def add_args(parser):
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e2)

    parser.add_argument("--batch_size", type=int, default=400)
    return parser.parse_args()


def wait_for_the_training_process():
    pipe_path = "/tmp/pipe_transformer_training_status_cv"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'\n" % message)
                print("Training is finished. Start the next training with...\n")
                return
            sleep(3)
            print("Daemon is alive. Waiting for the training result.")


# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

lr = [0.06, 0.01, 0.03, 0.001, 0.003, 0.1, 0.3]
freeze_strategies = ["mild", "start_from_freeze_all", "freeze_by_epoch"]
batch_size = [400]

os.system("kill $(ps aux | grep \"main_cv.py\" | grep -v grep | awk '{print $2}')")

run_id = 0

for bs_idx in range(len(batch_size)):
    for freeze_strategy in freeze_strategies:
        for lr_idx in range(len(lr)):
            current_lr, current_bs = lr[lr_idx], batch_size[bs_idx]
            args.lr = current_lr
            args.batch_size = current_bs
            args.run_id = run_id
            args.freeze_strategy = freeze_strategy
            logging.info("current_lr = %f, current_bs = %d, freeze_strategy = %s" % (current_lr, current_bs, freeze_strategy))

            os.system("nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 11122 1 \"ib0\""
                      " {args.lr} 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 {args.freeze_strategy}> "
                      "./PipeTransformer-imagenet-node0_r{args.run_id}.log 2>&1 &".format(args=args))

            wait_for_the_training_process()

            logging.info("cleaning the training...")
            os.system("kill $(ps aux | grep \"main_cv.py\" | grep -v grep | awk '{print $2}')")

            sleep(30)
            run_id += 1
