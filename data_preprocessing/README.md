# Optimizations for Data Loading
In standard data parallel-based distributed training, PyTorch uses DistributedSampler to make sure each worker in DP only load a subset of the original dataset that is exclusive to each other.
Compared to the standard strategy, we made following optimizations:
1. Dynamic partition: the number of DP workers is increased when new pipelines are participated in DP. 
In order to guarantee the data partition even after adding new pipes, the training dataset is repartitioned by rebuilding the DistributedSampler and setting new rank. 

2. To reuse the computation of FP for frozen layers, we cached the hidden states in host memory and disk memory as well. 
Since the training requires shuffle each epoch, the cache order of hidden features with respect to the order of original samples is different across different epochs.
In order to identify which data point a hidden feature belongs to, we build a sample unique ID by returning "index" in the get_item() function of Dataset class.
With this unique ID, we can find a sample's hidden feature with O(1) time complexity during training.

3. When data is shuffled in each epoch, a data sample trained in the previous epoch may be moved to another machine for training. 
This makes the cache not reused across epochs. To address this issue, we fix a subset of entire samples in a machine and only do shuffle for this subset.
This guarantees the shuffle during epochs is only executed inside a machine, thus hidden feature's cache can be reused deterministically.
To achieve this, rather than maintaining a global rank for DistributedSampler, we introduce `node_rank` and `local_rank`. 
`node rank` is used to identify which subset of samples a machine needs to hold. `local_rank` is used by DistributedSampler to identify which part of the shuffle subset that a worker inside a machine should train.
Note that this does not hurt the algorithmic convergence property, shuffling for multiple subsets obtains more randomness than a randomness obtained by a global shuffle, which further increases the robustness of training. The only difference is that some parallel processes in distributed training are fixed in part of the shuffled datasets. If a training task does not need to shuffle the dataset across epochs, the above mentioned optimization will not be activated.
