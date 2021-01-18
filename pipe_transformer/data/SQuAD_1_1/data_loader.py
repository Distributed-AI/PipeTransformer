import json
import os

from ..base.base_raw_data_loader import BaseRawDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "span_extraction"
        self.document_X = []
        self.question_X = []
        self.attributes = dict()
        self.train_file_name = "train-v1.1.json"
        self.test_file_name = "dev-v1.1.json"

    def data_loader(self):
        if len(self.document_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            context_X, question_X, Y, question_ids = self.process_data(
                os.path.join(self.data_path, self.train_file_name))
            train_size = len(context_X)
            temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
            context_X.extend(temp[0])
            question_X.extend(temp[1])
            Y.extend(temp[2])
            question_ids.extend(temp[3])
            train_index_list = [i for i in range(train_size)]
            test_index_list = [i for i in range(train_size, len(context_X))]
            index_list = train_index_list + test_index_list
            self.context_X, self.question_X, self.Y, self.question_ids = context_X, question_X, Y, question_ids
            self.attributes["train_index_list"] = train_index_list
            self.attributes["test_index_list"] = test_index_list
            self.attributes["index_list"] = index_list
        return {"context_X": self.context_X, "question_X": self.question_X, "Y": self.Y,
                "question_ids": self.question_ids,
                "attributes": self.attributes, "task_type": self.task_type}

    def process_data(self, file_path):
        context_X = []
        question_X = []
        Y = []
        question_ids = []
        if "doc_index" not in self.attributes:
            self.attributes["doc_index"] = []
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

            for doc_idx, document in enumerate(data["data"]):
                for paragraph in document["paragraphs"]:
                    for qas in paragraph["qas"]:
                        for answer in qas["answers"]:
                            context_X.append(paragraph["context"])
                            question_X.append(qas["question"])
                            start = answer["answer_start"]
                            end = start + len(answer["text"].rstrip())
                            Y.append((start, end))
                            question_ids.append(qas["id"])
                            self.attributes["doc_index"].append(doc_idx)

        return context_X, question_X, Y, question_ids


def get_normal_format(dataset, cut_off=None):
    """
    reformat the dataset to normal version.
    """
    reformatted_data = []
    assert len(dataset["context_X"]) == len(dataset["question_X"]) == len(dataset["Y"]) == len(dataset["question_ids"])
    for c, q, a, qid in zip(dataset["context_X"], dataset["question_X"], dataset["Y"], dataset["question_ids"]):
        item = {}
        item["context"] = c
        item["qas"] = [
            {
                "id": "%d" % (len(reformatted_data) + 1),
                "qid": qid,
                "is_impossible": False,
                "question": q,
                "answers": [{"text": c[a[0]:a[1]], "answer_start": a[0]}],
            }
        ]
        reformatted_data.append(item)
    return reformatted_data[:cut_off]
