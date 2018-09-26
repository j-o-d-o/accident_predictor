from abc import ABCMeta, abstractmethod


class IDataReader(metaclass=ABCMeta):

    @abstractmethod
    def get_nb_batches(self) -> int:
        """ get number of batches for one epoch """

    @abstractmethod
    def reset_epoch(self) -> None:
        """ restart epoch """

    @abstractmethod
    def get_next(self, mode="train") -> [list, list, bool]:
        """
            get next batch, mode should be able to handle 'train', 'validation' and 'test'
            returns batch_x, batch_y, finished
        """
