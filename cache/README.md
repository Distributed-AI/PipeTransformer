Two-level dynamic caching system for large scale

1. (done) Customized sampler, and sample ID mapping
2. (done) Shared memory to make sure the newly added processes can access the cache in host memory
3. (done) DiskCache with Pickle file
4. CacheManagerProcess: support two-level cache, handle the host memory and disk storage
7. CacheManagerProcess, Sliding window algorithm and mini-batch organization.

Shared_memory:
https://bugs.python.org/issue35813