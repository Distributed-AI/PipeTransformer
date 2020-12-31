import ctypes
import logging
import threading
import time
import traceback
from multiprocessing import Process

"""
PyTorch Multiprocessing and Queue:
https://github.com/pytorch/pytorch/blob/cd608fe59b70fa7cafb07110096b2e023a8b6e9c/test/test_multiprocessing.py#L583-L603

Python Multiprocessing and Queue:
https://stackoverflow.com/questions/11515944/how-to-use-multiprocessing-queue-in-python
"""


class DiskStorageProcess(Process):
    def __init__(self, queue, train_extracted_features_dict, test_extracted_features):
        super(DiskStorageProcess, self).__init__()
        self._stop_event = threading.Event()
        self.queue = queue

        self.train_extracted_features_dict = train_extracted_features_dict
        self.test_extracted_features = test_extracted_features
        self.iteration_num = 100

    def run(self):
        logging.debug("Starting " + self.name)
        while True:
            try:

                time.sleep(1)
            except Exception:
                traceback.print_exc()

    def cache_to_disk_storage(self):
        def _extract_features_impl(args, model, dataloader, is_train=True):
            model.eval()

            directory_train = "./extracted_features/" + args.dataset + "/"
            if is_train:
                path_train = directory_train + "train.pkl"
            else:
                path_train = directory_train + "test.pkl"

            if not os.path.exists(directory_train):
                os.makedirs(directory_train)

            chunks = 3

            train_data_extracted_features = dict()
            if path.exists(path_train):
                for chunk_index in range(chunks):
                    train_data_extracted_features = load_from_pickle_file(path_train)
            else:
                with torch.no_grad():
                    for batch_idx, (x, target) in enumerate(dataloader):
                        time_start_test_per_batch = time.time()
                        x = x.to(device)
                        extracted_feature_x, _, hidden_representations = model.transformer(x)
                        # for i, h in enumerate(hidden_representations):
                        #     hidden_representations[i] = h.cpu().detach()
                        train_data_extracted_features[batch_idx] = (extracted_feature_x.cpu().detach(), target)
                        time_end_test_per_batch = time.time()
                        logging.info("train_local feature extraction - time per batch = " + str(
                            time_end_test_per_batch - time_start_test_per_batch))

            save_as_pickle_file(path_train, train_data_extracted_features)
            return train_data_extracted_features
