from dlpipe.result import Result


class Callback:
    def training_start(self, result: Result) -> None:
        """ function is called once the training is starting"""

    def training_end(self, result: Result) -> None:
        """ function is called once the training finishes """

    def epoch_end(self, result: Result) -> None:
        """ function is called after each epoch """

    def batch_end(self, result: Result) -> None:
        """ function is called after each batch """

    def test_start(self, result: Result) -> None:
        """ function is called once the test is starting"""

    def test_end(self, result: Result) -> None:
        """ function is called after test """
