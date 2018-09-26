""" Data Reader for MongoDB """
import numpy as np
from typing import List, Tuple
from pymongo.collection import Collection
from random import shuffle

from dlpipe.data_reader.data_reader_base import BaseDataReader
from dlpipe.utils import DLPipeLogger


class MongoDBReader(BaseDataReader):
    def __init__(self,
                 collection: Collection,
                 batch_size: int = 32,
                 val_batch_size: int = None,
                 data_split: List[float] = list(),
                 processors: List[any] = list(),
                 shuffle_data: bool = True,
                 shuffle_steps: int = 1,
                 fields: List[str] = list(),
                 sort_by: Tuple = None,
                 limit: int= None):
        super().__init__(batch_size, val_batch_size, data_split, processors)
        self.collection = collection
        self.shuffle_data = shuffle_data
        self.shuffle_steps = shuffle_steps
        self.fields = fields
        self.sort_by = sort_by
        self.limit = limit

        self.last_index = {"train": 0, "validation": 0, "test": 0}
        self.doc_ids = {"train": [], "validation": [], "test": []}
        self.nb_docs = 0

        self._load_doc_ids()

    def _load_doc_ids(self):
        """ loading of all docIDs for the given connection and splitting them up in a train, validation and test set """
        DLPipeLogger.logger.info("Loading Document IDs from MongoDB")
        db_cursor = self.collection.find({}, {"_id": 1})
        if self.sort_by is not None:
            db_cursor.sort(self.sort_by)
        if self.limit:
            db_cursor.limit(self.limit)
        tmp_docs = []
        for doc in db_cursor:
            tmp_docs.append(doc["_id"])
        if self.shuffle_data:
            if self.shuffle_steps == 1:
                shuffle(tmp_docs)
            else:
                x = np.reshape(tmp_docs, (-1, self.shuffle_steps))
                np.random.shuffle(x)
                tmp_docs = x.flatten().tolist()

        self.nb_docs = len(tmp_docs)
        train_range = int(self.data_split[0] / 100 * self.nb_docs)
        va_range = int(train_range + self.data_split[1] / 100 * self.nb_docs)
        self.doc_ids["train"] = tmp_docs[:train_range]
        self.doc_ids["validation"] = tmp_docs[train_range:va_range]
        self.doc_ids["test"] = tmp_docs[va_range:]
        DLPipeLogger.logger.info("Documents loaded (train|validation|test): {0} | {1} | {2}\n\n".format(
            len(self.doc_ids["train"]), len(self.doc_ids["validation"]), len(self.doc_ids["test"])))

    def reset_epoch(self):
        """ Reset epoch by shuffling data and setting the index counters back to zero """
        if self.shuffle_data:
            shuffle(self.doc_ids["train"])

        self.last_index = {"train": 0, "validation": 0, "test": 0}

    def get_nb_batches(self) -> float:
        return len(self.doc_ids["train"]) / self.batch_size

    def _fetch_data(self, query_docs: list):
        """
        Get a set of _ids from the database (in order)
        :param query_docs: A list of _ids
        :return: A pymongo cursor
        """
        # to ensure the order of query_docs, use this method. For more details look at this stackoverflow question:
        # https://stackoverflow.com/questions/22797768/does-mongodbs-in-clause-guarantee-order/22800784#22800784
        query = [
            {"$match": {"_id": {"$in": query_docs}}},
            {"$addFields": {"__order": {"$indexOfArray": [query_docs, "$_id"]}}},
            {"$sort": {"__order": 1}}
        ]
        docs = self.collection.aggregate(query)
        return docs

    def _next_doc_ids(self, mode="train") -> [list, int, str]:
        """
        Get the next set of MongoDB _ids to fetch from the database for a certain mode
        :param mode: default="train", can be one of these ["train", "validation", "test"]
                     determines which data (train, validation, test) should be used
        :return: list of ids, last index of the doc_ids after getting this batch as int, a finished flag as bool
        """
        if mode != "train" and self.val_batch_size is None:
            return_ids = self.doc_ids[mode][:]
            return return_ids, 0, True
        else:
            batch_size = self.batch_size
            if mode != "train":
                batch_size = self.val_batch_size

            end_index = self.last_index[mode] + batch_size
            return_ids = self.doc_ids[mode][self.last_index[mode]:end_index]
            # check if the next batch has still enough values for a full batch, if not -> set finished = True
            finished = (end_index + batch_size) >= len(self.doc_ids[mode])
            return return_ids, end_index, finished

    def get_next(self, mode: str="train"):
        """
        Returns data for the next docId for a certain mode and starts over if the end of the data is reached
        :param mode: default="train", can be one of these ["train", "validation", "test"]
                     determines which data (train, validation, test) should be used
        :returns: array of 3 values with: [batch data input, batch data ground truth, finished flag]
        """
        assert mode in ["train", "validation", "test"]

        next_doc_ids, end_index, finished = self._next_doc_ids(mode)
        doc_list = list(self._fetch_data(next_doc_ids))
        batch_x, batch_y = self._process_batch(doc_list)

        if finished:
            self.last_index[mode] = 0
            if self.shuffle_data:
                shuffle(self.doc_ids[mode])
        else:
            self.last_index[mode] = end_index

        return np.asarray(batch_x), np.asarray(batch_y), finished
