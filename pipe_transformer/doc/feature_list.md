# System-wise Optimization:
System-wise benefits:

1. computation can be reused
2. communication is not required
3. memory cost is reduced

1. DPipe: a pytorch compatible pipe parallelism for large model like BERT, Vision Transformer.

2. Pipe Load Balance: making the computation inside pipe evenly

3. Elastic Pipelining: adding more pipes to increase parallelism when memory is released.

4. Automated DP:
(1) make the newly created parallel processes in DP can communicate with the existing DP processes.
(2) skip parameters during cross-pipe synchronization, largely reducing the communication cost.

5. Dynamic Cache: 
Two-level shared memory: each frozen layer only computes the forward propagation once!

https://docs.python.org/3/library/multiprocessing.shared_memory.html

We use multiprocessing because it can avoid GIL (global interpreter lock) issue in python multi-threading

6. When the cache is responsible for a lot of frozen layers in the later stage, because the fp time is much longer than bp.
Overlap frozen bp and pipe bubble are suitable for the time when the fp time in the early stage is small, or when it is equivalent to the bp time

(5 and 6 are combination of two swords)

7. Model-size adaptive: depending on the model size, adaptively choose DP or Pipe in a single machine.
(1) Single GPU
(2) Single Machine, Multiple GPUs
(3) Multiple Machines, Multiple GPUs

(***) 8. What if we freeze randomly, does the system design still work?


# Machine Learning-wise Optimization

1. checking point for Optimizer, Scheduler

2. redistribute the dataset to additional pipes

3. auto freeze algorithm (*)


# Other Ideas
Distributed Training:
1. Elastic NAS: Large scale Distributed Neural Architecture Search with Pipe Transformer
2. Large scale Distributed Training for Video Action Recognition

FedML:
1. FedBERT: Efficient Training of BERT model for Cross-silo Federated Learning
2. Efficient and Personalized Federated Neural Architecture Search: Bridging the Accuracy Gap in Federated Learning

Destory CUDA context:

https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e

https://github.com/pytorch/pytorch/issues/37664

This idea can also be used in Federated Learning. We can calculate the frozen number at the server side.

https://arxiv.org/pdf/1706.05806.pdf

# GPipe
torch.__version__
'1.8.0.dev20201219