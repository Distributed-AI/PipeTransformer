# CIFAR 10

1. No Freeze
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 0 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 0 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
GPUs: 4 GPUs/machine x 1 \
Time Cost: 56 minutes \
Accuracy: 98.74% \
https://wandb.ai/automl/pipe_and_ddp/runs/24qpexyd 

GPUs: 8 GPUs/machine x 2 \
Time Cost: 24 minutes \
Accuracy: 

2. Freeze (Auto) + No Cache
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
GPUs: 8 GPUs/machine x 2 \
Time Cost: 20 minutes \
Accuracy: 98.6% (98.74%) \
https://wandb.ai/automl/pipe_and_ddp/runs/cb1dx5e4?workspace=user-chaoyanghe-com



3. Freeze (Auto) + One-level Cache
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
GPUs: 8 GPUs/machine x 2 \
Time Cost: 
Accuracy: 

3. Freeze (Auto) + Shared memory two-level Cache
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
GPUs: 8 GPUs/machine x 2 \
Time Cost: 
Accuracy: 

# CIFAR 100
1. No Freeze
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.1 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.1 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
GPUs: 8 GPUs/machine x 2
Time Cost: 24.5 minutes
Accuracy: 91.66%
Experimental Results: https://wandb.ai/automl/pipe_and_ddp/runs/18tm5xsr

2. Freeze (handcrafted)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
Time Cost: 12 minutes


2.Freeze (AutoFreeze Algorithm)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
Time Cost: ? minutes

# ImageNet. Batch Size = 400

1.Freeze (handcrafted)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 11122 1 "ib0" 0.06 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 11122 1 "ib0" 0.06 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &

```
Time cost: 

2.Freeze (AutoFreeze Algorithm)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 11122 1 "ib0" 0.06 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 11122 1 "ib0" 0.06 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &
```
Time cost: