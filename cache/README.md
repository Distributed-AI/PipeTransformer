Two-level dynamic caching system for large scale

1. (done) Customized sampler, and sample ID mapping
2. (done) Shared memory to make sure the newly added processes can access the cache in host memory
3. (done) DiskCache with Pickle file
4. CacheManagerProcess: support two-level cache, handle the host memory and disk storage
7. CacheManagerProcess, Sliding window algorithm and mini-batch organization.

Shared_memory:
https://bugs.python.org/issue35813

Debugging Experience (in Chinese):

1. pipe transformer后需要让新增process更新到最新的weights，包括frozen layers, pipe model。否则会使用过时weights产生新的gradient。
2. shared_memory的操作需要原子性保证
3. 为了或许O(1)查找效率，进行了sample level的tensor缓存，但是读取时用于训练时按照batch level的使用，有一个for loop a batch的操作，这个操作必须确保每个sample tensor的唯一性，不能混入其他worker的写入操作，因此我给cache做了一个唯一id: sample_id + frozen_layer_id