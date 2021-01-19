from abc import ABC, abstractmethod


class BaseDataManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_data_loader_with_node_rank(self, epoch, batch_size, node_rank, num_replicas, local_rank):
        pass

    @abstractmethod
    def get_train_sample_index(self, epoch):
        pass

    @abstractmethod
    def get_test_sample_index(self, epoch):
        pass
