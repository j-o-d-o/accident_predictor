from dlpipe.callbacks import Callback
from dlpipe.schemas import ExperimentSchema
from dlpipe.utils import DLPipeLogger


class SaveExpMongoDB(Callback):
    """
    Callback class to save keras models during training, usually after each epoch
    or if self._epoch_save_condition(result) returns true. Append this Callback to the Trainer e.g.:

    >> callback = SaveExpMongoDB(model_db, "my_model_name", model.get_config())
    >> trainer = Trainer(model=model, data_reader=data_reader, callbacks=[callback])

    """
    def __init__(
            self,
            mongo_db,
            name,
            keras_model,
            save_initial_weights: bool=True,
            epoch_save_condition=None):
        self._epoch_save_condition = epoch_save_condition
        self._save_initial_weights = save_initial_weights
        self._db = mongo_db
        self._collection = mongo_db["experiment"]
        self._keras_model = keras_model
        self._exp = ExperimentSchema(self._collection, name, keras_model)
        self._exp.log_file_path = DLPipeLogger.get_log_file_path()
        self._exp.save()

    def get_exp_id(self):
        return self._exp.id

    def training_start(self, result):
        self._exp.result = result
        self._exp.status = 100
        self._exp.update(update_result=self._save_initial_weights)

    def batch_end(self, result):
        self._exp.result = result
        self._exp.update(update_result=False)

    def epoch_end(self, result):
        self._exp.result = result
        should_save_weights = self._epoch_save_condition is None or self._epoch_save_condition(result)
        self._exp.update(update_result=should_save_weights)

    def training_end(self, result):
        self._exp.status = 2
        self._exp.update(update_result=False)

    def test_start(self, result):
        self._exp.status = 200
        self._exp.update(update_result=False)

    def test_end(self, result):
        self._exp.status = 1
        # no new weights to save after testing, just metrics
        self._exp.update(update_result=True, update_weights=False)
