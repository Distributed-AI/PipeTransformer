import os
import pickle
import sys

import torch


class DiskMemoryManager:
    def __init__(self, name):
        self.name = name

    def set(self, sample_uid, layer_id, hidden_tensor):
        file_path = self._build_path(sample_uid, layer_id)
        hidden_np = hidden_tensor.numpy()
        pickle.dump(hidden_np, open(file_path, "wb"))

    def get(self, sample_uid, layer_id):
        file_path = self._build_path(sample_uid, layer_id)
        hidden_np = pickle.load(open(file_path, "rb"))
        tensor = torch.from_numpy(hidden_np).cpu()
        return tensor

    def delete(self, sample_uid, layer_id):
        file_path = self._build_path(sample_uid, layer_id)
        if os.path.exists(file_path):
            print("delete")
            os.remove(file_path)

    def _build_path(self, sample_uid, layer_id):
        path_level1 = sample_uid // 100 + 1
        path_level2 = path_level1 // 100 + 1
        path_level3 = path_level2 // 100 + 1
        cache_path = os.path.join("./.cache", str(path_level3 % 100), str(path_level2 % 100),
                                  str(path_level1 % 100))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
        file_path = cache_path + "/" + self.name + "_" + str(sample_uid) + "_" + str(layer_id) + ".fea"
        print(file_path)
        return file_path


if __name__ == "__main__":
    disk_cache = DiskMemoryManager("hidden_feature")
    blocks = 2000
    for i in range(blocks):
        hidden_tensor_set = torch.randn([500, 197, 768])
        disk_cache.set(i, i, hidden_tensor_set)

        hidden_tensor = disk_cache.get(i, i)
        if torch.equal(hidden_tensor, hidden_tensor_set):
            print("equal")

    print("************")
    # for i in range(blocks):
    #     disk_cache.delete(i, i)
