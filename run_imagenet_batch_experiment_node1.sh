# nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-CIFAR100-node1.log 2>&1 &
sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.03 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1

sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.01 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1

sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.3 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1

# nohup sh
# nohup sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.1 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1 > ./PipeTransformer-imagenet-node1.log 2>&1 &
run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.1 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1

sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.003 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1

sh run_elastic_pipe.sh 8 2 1 192.168.11.2 22222 1 "ib0" 0.001 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 1