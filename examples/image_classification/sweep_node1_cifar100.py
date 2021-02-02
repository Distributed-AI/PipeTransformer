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

lr = [0.03, 0.01, 0.1, 0.3]
freeze_strategies = ["linear"]
batch_size = [320]
# freeze_hpo = ["freeze", "no_freeze"]
freeze_hpo = ["freeze"]
# autopipe_hpo = ["auto_pipe", "no_auto_pipe"]
autopipe_hpo = ["auto_pipe"]
# autodp_hpo = ["auto_dp", "no_auto_dp"]
autodp_hpo = ["auto_dp"]
# autocache_hpo = ["cache", "no_cache"]
autocache_hpo = ["cache"]

freeze_strategy_alpha_hpo = [0.2, 0.3, 0.4, 0.5]

os.system("kill $(ps aux | grep \"main_cv.py\" | grep -v grep | awk '{print $2}')")

run_id = 0

for bs_idx in range(len(batch_size)):
    for freeze_strategy in freeze_strategies:
        for lr_idx in range(len(lr)):
            for auto_freeze in freeze_hpo:
                for autopipe in autopipe_hpo:
                    for autodp in autodp_hpo:
                        for autocache in autocache_hpo:
                            for freeze_strategy_alpha in freeze_strategy_alpha_hpo:
                                current_lr, current_bs = lr[lr_idx], batch_size[bs_idx]
                                args.lr = current_lr
                                args.batch_size = current_bs
                                args.run_id = run_id
                                args.freeze_strategy = freeze_strategy
                                args.auto_freeze = auto_freeze
                                args.autopipe = autopipe
                                args.autodp = autodp
                                args.autocache = autocache
                                args.freeze_strategy_alpha = freeze_strategy_alpha
                                logging.info("current_lr = %f, current_bs = %d, freeze_strategy = %s" % (current_lr, current_bs, freeze_strategy))

                                # sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 ib0 0.03 320 cifar100 ./../../data/cifar100/ 8 no_freeze no_auto_pipe no_auto_dp no_cache
                                os.system("nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 \"ib0\""
                                          " {args.lr} 320 cifar100 ./../../data/cifar100/ 8 {args.freeze_strategy_alpha} {args.auto_freeze} {args.autopipe} {args.autodp} {args.autocache} > "
                                          "./PipeTransformer-cifar100-node0_r{args.run_id}.log 2>&1 &".format(args=args))

                                wait_for_the_training_process()

                                logging.info("cleaning the training...")
                                os.system("kill $(ps aux | grep \"main_cv.py\" | grep -v grep | awk '{print $2}')")

                                sleep(30)
                                run_id += 1
