import os

from nltk.tree import Tree

from ..base.base_raw_data_loader import BaseRawDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "text_classification"
        self.target_vocab = None
        self.train_file_name = "train.txt"
        self.test_file_name = "test.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
            train_size = len(X)
            temp_X, temp_Y = self.process_data(os.path.join(self.data_path, self.test_file_name))
            X.extend(temp_X)
            Y.extend(temp_Y)
            self.X, self.Y = X, Y
            train_index_list = [i for i in range(train_size)]
            test_index_list = [i for i in range(train_size, len(X))]
            index_list = train_index_list + test_index_list
            self.attributes = {"index_list": index_list, "train_index_list": train_index_list,
                               "test_index_list": test_index_list}
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    def label_level(self, label):
        return {'0': 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': 'positive', None: None}[label]

    def process_data(self, file_path):
        X = []
        Y = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                tree = Tree.fromstring(line)
                label = self.label_level(tree.label())
                if label != "neutral":
                    X.append(" ".join(tree.leaves()))
                    Y.append(label)
        return X, Y
