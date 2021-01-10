from cache.shared_memory_dict import SharedMemoryDict
from multiprocessing.shared_memory import SharedMemory

smd = SharedMemoryDict(name='hidden_feature1', size=1024)

sample_id = 100
print(smd.keys())
print(smd.get(sample_id).get_layer_id())
print(smd.get(sample_id).get_np_hidden_feature())

