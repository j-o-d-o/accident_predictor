"""
Trainer class to train keras models, when adding other training frameworks or methods, this might need some reworking
for better generalization
"""

import numpy as np
import math
from typing import List
from dlpipe.data_reader.data_reader_interface import IDataReader
from keras.models import Model
from dlpipe.result import Result
from dlpipe.utils import DLPipeLogger


class Trainer:
    def __init__(self, model: Model = None, data_reader: IDataReader = None, callbacks: List[any] = None):
        self._callbacks: List[any] = []
        self._print_counter: int = 0  # to make sure the console does not get spamed
        self._max_prints: int = 5
        self.data_reader: IDataReader = data_reader

        if callbacks is not None:
            self._callbacks = callbacks

        self.model = model
        self.result = Result()

    def set_model(self, model: Model):
        self.model = model

    def _create_metrics(self, results):
        metric_values = []
        if isinstance(results, (list, )):
            for i, metric_name in enumerate(self.model.metrics_names):
                metric_values.append({
                    "name": metric_name,
                    "value": results[i]
                })
        else:
            metric_values.append({
                "name": "loss",
                "value": results
            })
        return metric_values

    def calc_max_batch_size(self):
        # find least amount of batches
        max_batch_size = self.data_reader.get_nb_batches()
        return math.floor(max_batch_size)

    def _batch_printer(self, curr_epoch, curr_batch, epochs, results):
        if self.result.max_batches_per_epoch is None:
            self.result.max_batches_per_epoch = self.calc_max_batch_size()
            self.result.max_epochs = epochs

        percentage = ((curr_epoch * self.result.max_batches_per_epoch + curr_batch + 1) /
                      (epochs * self.result.max_batches_per_epoch))*100

        batch_percentage = (curr_batch + 1) / self.result.max_batches_per_epoch

        print_flag = self._print_counter <= math.floor(batch_percentage * self._max_prints)

        if print_flag:
            self._print_counter += 1
            display = "{0:.2f}% => \tEpoch: {1}\t".format(percentage, curr_epoch)
            metrics = self._create_metrics(results)
            for metric in metrics:
                display += "{0}: {1:.4f} \t".format(metric["name"], metric["value"])

            DLPipeLogger.logger.info(display)

    def _validation(self, curr_epoch):
        val_finished = False
        val_counter = 0
        tmp_val_results = []
        while not val_finished:
            x, y, val_finished = self.data_reader.get_next(mode="validation")
            tmp_val_results.append(self._create_metrics(self.model.test_on_batch(x, y)))
            val_counter += 1

        # TODO: make averages for multiple results in case val_batch_size is set on the reader (val_counter > 1)
        final_results = tmp_val_results[0]
        for metric_result in final_results:
            self.result.append_to_metric(metric_result["name"], metric_result["value"], phase="validation")

        display = "Validation => \tEpoch: {0}\t".format(curr_epoch)
        for metric in final_results:
            display += "{0}: {1:.4f} \t".format(metric["name"], metric["value"])
        DLPipeLogger.logger.info(display+"\n")

    def train(self, epochs: int = 5, sample_weight=None, class_weight=None):
        current_epoch = 0
        current_batch = 0
        finished = False

        # at epoch -1 the weights are set to initialized weights
        self.result.update_weights(self.model)

        for cb in self._callbacks:
            cb.training_start(self.result)

        while not finished:
            epoch_finished = False
            batches = []
            x, y, dr_finished = self.data_reader.get_next(mode="train")
            batches.append({
                "x": x,
                "y": y
            })
            if dr_finished:
                epoch_finished = True

            # for now just take the data of the first data_reader
            input_data = np.asarray(batches[0]["x"])
            ground_truth = np.asarray(batches[0]["y"])

            # Train the model
            results = self.model.train_on_batch(input_data, ground_truth,
                                                sample_weight=sample_weight, class_weight=class_weight)

            # update result instance for the training results
            self.result.update_weights(self.model, current_epoch, current_batch)
            for i, metric_result in enumerate(results):
                self.result.append_to_metric(self.model.metrics_names[i], metric_result, phase="training")

            for cb in self._callbacks:
                cb.batch_end(self.result)

            self._batch_printer(current_epoch, current_batch, epochs, results)

            if epoch_finished:
                self._print_counter = 0
                self._validation(current_epoch)

                self.data_reader.reset_epoch()

                for cb in self._callbacks:
                    cb.epoch_end(self.result)

                current_batch = 0
                current_epoch += 1
            else:
                current_batch += 1

            if current_epoch >= epochs:
                finished = True

        for cb in self._callbacks:
            cb.training_end(self.result)

    def test(self):
        test_finished = False
        val_counter = 0
        tmp_val_results = []
        while not test_finished:
            x, y, test_finished = self.data_reader.get_next(mode="test")
            tmp_val_results.append(self._create_metrics(self.model.test_on_batch(x, y)))
            val_counter += 1

        # TODO: make averages for multiple results in case val_batch_size is set on the reader (val_counter > 1)
        final_results = tmp_val_results[0]
        for metric_result in final_results:
            self.result.append_to_metric(metric_result["name"], metric_result["value"], phase="test")

        display = "Test => \t"
        for metric in final_results:
            display += "{0}: {1:.4f} \t".format(metric["name"], metric["value"])
        DLPipeLogger.logger.info(display)

        for cb in self._callbacks:
            cb.test_end(self.result)

        return final_results
