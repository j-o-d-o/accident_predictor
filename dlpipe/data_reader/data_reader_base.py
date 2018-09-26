"""
Base class for any DataReader
"""
from typing import List
from dlpipe.data_reader.data_reader_interface import IDataReader


class BaseDataReader(IDataReader):
    def __init__(self,
                 batch_size: int = 32,
                 val_batch_size: int = None,
                 data_split: List[float] = list(),
                 processors: List[any] = list()):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.processors = processors

        assert(len(data_split) == 3 and sum(data_split) == 100)
        self.data_split = data_split

    def add_processors(self, processors: list):
        self.processors.extend(processors)

    def _process_batch(self, batch):
        """
        Process the raw data from the data reader with the specified processors
        :param batch: raw data for this batch (must be iterable)
        :return: array of 2: [batch data input, batch data ground truth]
        """
        batch_x = []
        batch_y = []
        for data in batch:
            input_data = None
            ground_truth = None
            piped_params = {}
            raw_data = data
            # process each entry in the batch list one by one
            for processor in self.processors:
                raw_data, input_data, ground_truth, piped_params = processor.process(raw_data, input_data, ground_truth,
                                                                                     piped_params=piped_params)
            batch_x.append(input_data)
            batch_y.append(ground_truth)
        return batch_x, batch_y

    def get_nb_batches(self) -> int:
        raise NotImplementedError("get_nb_batches() must be implemented by the the DataReader")

    def reset_epoch(self) -> None:
        raise NotImplementedError("reset_epoch() must be implemented by the the DataReader")

    def get_next(self, mode="train") -> [list, list, bool]:
        raise NotImplementedError("get_next() must be implemented by the the DataReader")
