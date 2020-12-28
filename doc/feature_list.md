# System-wise Optimization:

1. DPipe: a pytorch compatible pipe parallelism for large model like BERT, Vision Transformer.

2. Pipe Load Balance: making the communication inside pipe evenly

3. Elastic Pipelining: adding more pipes to increase parallelism when memory is released.

4. Automated DP:
(1) make the newly created parallel processes in DP can communicate with the existing DP processes.
(2) skip parameters during cross-pipe synchronization, largely reducing the communication cost.

5. Forward Cache

6. Depending on the model size, adaptively choose DP or Pipe in a single machine.

Machine Learning-wise Optimization:

1. checking point for Optimizer, Scheduler

2. redistribute the dataset to additional pipes

3. auto freeze algorithm