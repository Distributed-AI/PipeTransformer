

## Installation

1.create conda environment
```
conda create --name pipe_distributed python=3.7.4
conda activate pipe_distributed
```

2.install the latest nightly Pytorch, in which DPipe (torchpipe) is supported.

```
torch.__version__
'1.8.0.dev20201219
conda install pytorch==1.8.0.dev20201219 torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly


```

3.install other packages
```
pip install -r requirements.txt 
```

4. WandB.com
```
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```


## Pipe and DDP 
```
nohup sh run.sh 1 2 0 192.168.11.1 11111 1 0.03 256 cifar10 ./data/cifar10/ > ./machine1.txt 2>&1 &
nohup sh run.sh 1 2 1 192.168.11.1 11111 1 0.03 256 cifar10 ./data/cifar10/ > ./machine2.txt 2>&1 &

sh run.sh 1 2 0 192.168.11.1 11111 1
sh run.sh 1 2 1 192.168.11.1 11111 1

sh run.sh 1 2 0 192.168.1.1 11111 0
sh run.sh 1 2 1 192.168.1.1 11111 0

# ddp demo
sh run_ddp.sh 8 2 0 192.168.11.1 11111

# sh run_ddp.sh 8 2 0 192.168.11.1 11111
nohup sh run_ddp.sh 8 2 0 192.168.11.1 11111 > ./machine1.txt 2>&1 &
nohup sh run_ddp.sh 8 2 1 192.168.11.1 11111 > ./machine2.txt 2>&1 &

```

## Pipe and DDP (Elastic)

debug at 4GPUs:
```
nohup sh run_elastic_pipe.sh 4 1 0 192.168.1.73 11111 0 "wlx9cefd5fb3821" 0.03 60 cifar10 ./data/cifar10/ 4 1 > ./PipeTransformer-CIFAR10-freeze.log 2>&1 &

sh run_elastic_pipe.sh 4 1 0 192.168.1.73 22222 0 "wlx9cefd5fb3821" 0.03 120 imagenet /home/chaoyanghe/sourcecode/dataset/cv/ImageNet 4 1


```
CIFAR 10
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
Result:

CIFAR 100
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```

ImageNet. Batch Size = 320
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &
```

```
nohup sh run_cifar100_batch_experiment_node0.sh > ./PipeTransformer-cifar100-node0.log 2>&1 &
nohup sh run_cifar100_batch_experiment_node1.sh > ./PipeTransformer-cifar100-node1.log 2>&1 &
```

## DDP 
```
nohup sh run_ddp.sh 4 2 0 192.168.11.1 11111 1 0.03 64 cifar10 ./data/cifar10/ > ./machine1_ddp.txt 2>&1 &
nohup sh run_ddp.sh 4 2 1 192.168.11.1 11111 1 0.03 64 cifar10 ./data/cifar10/ > ./machine2_ddp.txt 2>&1 &

sh run.sh 1 2 0 192.168.11.1 11111 1
sh run.sh 1 2 1 192.168.11.1 11111 1

sh run.sh 1 2 0 192.168.1.1 11111 0
sh run.sh 1 2 1 192.168.1.1 11111 0

# ddp demo
sh run_ddp.sh 8 2 0 192.168.11.1 11111

# sh run_ddp.sh 8 2 0 192.168.11.1 11111
nohup sh run_ddp.sh 8 2 0 192.168.11.1 11111 > ./machine1.txt 2>&1 &
nohup sh run_ddp.sh 8 2 1 192.168.11.1 11111 > ./machine2.txt 2>&1 &

```

```
# kill all processes
kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
```

# Single GPU training (for performance evaluation)
```
sh run_single_gpu.sh 0.03 64 cifar10 ./data/cifar10/ 1
```



## Experimental Resuts
A single Pipe is on 3 GPUs / machine, DDP on 2 machines.

| Training Algorithm      | Chunks      | Batch Size     | Memory Cost Per GPU    | Images/Second    | Training Time    |
| :-------------| :------------- | :----------: | -----------: | -----------: | -----------: |
| DDP |  -  | 128  | OOM    | - | - |
| DDP |  -  | 64  | 14G    | 260 | 85 minutes |
| Pipe+DDP |  4  | 256  | OOM    | - | - |
| Pipe+DDP |  4  | 240  | 15G    | 294 | 77 minutes |

A single Pipe is on 4 GPUs / machine, DDP on 2 machines.

| Training Algorithm      | Chunks      | Batch Size     | Memory Cost Per GPU    | Images/Second    | Training Time    |
| :-------------| :------------- | :----------: | -----------: | -----------: | -----------: |
| DDP |  -  | 64  | OOM    | - | - |
| DDP |  -  | 60  | 16G    | 416 | 58m |
| Pipe+DDP | 8 | 330  |   OOM  | - |  -|
| Pipe+DDP | 8 | 320  |   15.5G  | 326 |  61 minutes|


When batch size is 64 and all transformer blocks are frozen, the forward propagation time is 470ms
```
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:243] INFO data loading time cost (ms) by CUDA event 3.2522239685058594
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:244] INFO forward time cost (ms) by CUDA event 471.1454772949219
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:245] INFO backwards time cost: (ms) by CUDA event 8.642560005187988
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:248] INFO sample_num_throughput (images/second): 200
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:251] INFO communication frequency (cross machine sync/second): 1.569294
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:253] INFO size_params_ddp_sum: 0.000000
 - Wed, 23 Dec 2020 21:42:13 main_elastic_pipe.py[line:195] INFO epoch = 0, batch index = 8/391
 
```

When batch size is 320 and all transformer blocks are frozen, the forward propagation time is 2300ms
```
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:243] INFO data loading time cost (ms) by CUDA event 15.884320259094238
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:244] INFO forward time cost (ms) by CUDA event 2313.234375
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:245] INFO backwards time cost: (ms) by CUDA event 31.178239822387695
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:248] INFO sample_num_throughput (images/second): 240
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:251] INFO communication frequency (cross machine sync/second): 0.375755
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:253] INFO size_params_ddp_sum: 0.000000
 - Wed, 23 Dec 2020 21:41:13 main_elastic_pipe.py[line:195] INFO epoch = 0, batch index = 8/79
```

Can be accelerated by multi-threading?
https://github.com/pytorch/pytorch/issues/22260

## Measurement
How to measure DDP time breakdown

https://discuss.pytorch.org/t/how-to-measure-ddp-time-breakdown/78925/2

https://discuss.pytorch.org/t/distributed-training-slower-than-dataparallel/81539/4

1. data loading time
2. forward propagation time

3. back propagation time

4. All_reduce time cost 
 1) change the reducer.cpp, and build from source
 2) nvprof

5. real time bandwidth cost

https://www.tecmint.com/nethogs-monitor-per-process-network-bandwidth-usage-in-real-time/

https://www.tecmint.com/linux-network-bandwidth-monitoring-tools/

https://www.debugpoint.com/2016/10/3-best-command-line-tool-network-bandwidth-monitoring-ubuntu-linux/