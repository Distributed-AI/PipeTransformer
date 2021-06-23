## Experiments
### CIFAR-100 + PipeTransformer (AutoFreeze + AutoPipe + AutoDP + AutoCache)
```
# run in node 0
sh run_pipetransformer.sh 8 2 0 10.0.198.185 1111 config/train_config_cifar100_all_in_one.yaml 1000

# run in node 1
sh run_pipetransformer.sh 8 2 1 10.0.198.185 1111 config/train_config_cifar100_all_in_one.yaml 1001
```

### CIFAR-100 No Freeze
```
# run in node 0
sh run_pipetransformer.sh 8 2 0 10.0.198.185 1111 config/train_config_cifar100_no_freeze.yaml 1002

# run in node 1
sh run_pipetransformer.sh 8 2 1 10.0.198.185 1111 config/train_config_cifar100_no_freeze.yaml 1003
```

```
# kill all processes
kill $(ps aux | grep "main_cv.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "main_tc.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "main_qa.py" | grep -v grep | awk '{print $2}')
```



## Development Notes

Can be accelerated by multi-threading?
https://github.com/pytorch/pytorch/issues/22260

How to measure DDP time breakdown:

https://discuss.pytorch.org/t/how-to-measure-ddp-time-breakdown/78925/2

https://discuss.pytorch.org/t/distributed-training-slower-than-dataparallel/81539/4

1. data loading time
2. forward propagation time

3. back propagation time

4. All_reduce time cost: 1) change the reducer.cpp, and build from source 2) nvprof

5. real time bandwidth cost

https://www.tecmint.com/nethogs-monitor-per-process-network-bandwidth-usage-in-real-time/

https://www.tecmint.com/linux-network-bandwidth-monitoring-tools/

https://www.debugpoint.com/2016/10/3-best-command-line-tool-network-bandwidth-monitoring-ubuntu-linux/


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

