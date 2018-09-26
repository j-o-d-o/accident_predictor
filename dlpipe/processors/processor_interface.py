from abc import ABCMeta, abstractmethod


class IPreProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        ...
        return raw_data, input_data, ground_truth, piped_params
