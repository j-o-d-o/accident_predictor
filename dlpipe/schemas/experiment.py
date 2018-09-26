"""
Data Container for an Experiment (which also saves it to the mongodb)
"""
from dlpipe.result import Result
from bson import ObjectId
import gridfs
import os


class ExperimentSchema:
    def __init__(self,
                 collection,
                 name: str,
                 keras_model,
                 ):
        # model info
        self.keras_model = keras_model
        self.name: str = name
        self.result: Result = None
        # training info
        self.status: int = 0
        self.log_file_path = ""
        self.id = None
        # mongodb connection
        self._collection = collection

    def get_dict(self) -> dict:
        """
        :return: dict of serialized experiment data
        """
        return_dict = {
            "name": self.name,
            "keras_model": self.keras_model,
            "status": self.status,
            "log_file_path": self.log_file_path,
            "curr_epoch": None,
            "curr_batch": None,
            "max_batches_per_epoch": None,
            "max_epochs": None
        }
        if self.result is not None:
            return_dict.update({
                "curr_epoch": self.result.curr_epoch,
                "curr_batch": self.result.curr_batch,
                "max_batches_per_epoch": self.result.max_batches_per_epoch,
                "max_epochs": self.result.max_epochs
            })
        return return_dict

    def save(self):
        data_dict = self.get_dict()
        data_dict["metrics"] = None
        data_dict["weights"] = []
        if self._collection is not None:
            self.id = self._collection.insert_one(data_dict).inserted_id
            self.update_result()

    def update(self, update_result: bool=True, update_weights: bool=True):
        data_dict = self.get_dict()
        if self._collection is not None:
            self._collection.update_one(
                {'_id': ObjectId(self.id)},
                {
                    '$set': data_dict
                }
            )
        if update_result:
            self.update_result(update_weights=update_weights)

    def update_result(self, update_weights: bool=True):
        if self.result is not None and self._collection is not None:
            if update_weights:
                fs = gridfs.GridFS(self._collection.database)
                tmp_filename = "tmp_model_weights_save.h5"
                model_gridfs = None
                if self.result.model is not None:
                    self.result.model.save(tmp_filename)
                    with open(tmp_filename, mode='rb') as file:
                        file_bytes = file.read()
                        model_gridfs = fs.put(file_bytes)
                    os.remove(tmp_filename)

                weights = {
                    "model_gridfs": model_gridfs,
                    "epoch": self.result.curr_epoch,
                    "batch": self.result.curr_batch
                }
                query = {
                    '$set': {
                        'metrics': self.result.metrics,
                    },
                    '$push': {'weights': weights}
                }
            else:
                query = {
                    '$set': {
                        'metrics': self.result.metrics,
                    }
                }

            self._collection.update_one(
                {'_id': ObjectId(self.id)},
                query
            )
