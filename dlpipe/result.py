"""
Container class for the results of the model
TODO: find another way to save weight independent of the usage of keras models
"""


class Result:
    def __init__(self):
        self.metrics = {
            "training": {},
            "validation": {},
            "test": {}
        }
        self.model = None
        self.max_batches_per_epoch: int = None
        self.max_epochs: int = None
        self.curr_epoch: int = -1  # -1 represents initialization
        self.curr_batch: int = 0

    def append_to_metric(self, metric_name: str, value: any, phase: str="training", epoch: int=None, batch: int=None):
        if phase not in self.metrics:
            raise ValueError("phase must be any of " + str(self.metrics.keys()))
        if epoch is None:
            epoch = self.curr_epoch
        if batch is None:
            batch = self.curr_batch
        if metric_name not in self.metrics[phase]:
            self.metrics[phase][metric_name] = []
        self.metrics[phase][metric_name].append({"value": float(value), "epoch": epoch, "batch": batch})

    def update_weights(self, model, curr_epoch: int=None, curr_batch: int=None):
        if curr_epoch is None:
            curr_epoch = self.curr_epoch
        if curr_batch is None:
            curr_batch = self.curr_batch
        self.model = model
        self.curr_epoch = curr_epoch
        self.curr_batch = curr_batch
