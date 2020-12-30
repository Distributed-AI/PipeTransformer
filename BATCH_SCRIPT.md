# CIFAR 10
1.Freeze (handcrafted)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
Time Cost: 12 minutes

```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 0 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 0 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```
Time Cost: ? minutes

# CIFAR 100
1.Freeze (handcrafted)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
Time Cost: 12 minutes


2.Freeze (AutoFreeze Algorithm)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
Time Cost: ? minutes

# ImageNet. Batch Size = 400

1.Freeze (handcrafted)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &
```
Time cost: 

2.Freeze (AutoFreeze Algorithm)
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 0 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 0 > ./PipeTransformer-imagenet-node1.log 2>&1 &
```
Time cost: