CIFAR 10
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar10 ./data/cifar10/ 8 1 > ./PipeTransformer-CIFAR10-node1.log 2>&1 &
```

CIFAR 100
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 cifar100 ./data/cifar100/ 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
```
Freeze Time cost: 12 minutes
No Freeze Time cost: ? minutes

ImageNet. Batch Size = 320
```
nohup sh run_elastic_pipe.sh 8 2 0 192.168.11.2 22222 1 0.03 320 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node0.log 2>&1 &
nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 0.03 320 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &
```

