import logging
import io
import os


class DLPipeLogger:

    _logger_level = logging.DEBUG
    _logger_name = 'DLPipe.logger'
    _formatter = logging.Formatter('[%(asctime)s] %(levelname)-10s %(message)s')
    _log_contents = io.StringIO()
    _current_log_file_path = "dlpipe.log"
    logger = None
    string_handler = None
    file_handler = None
    console_handler = None

    @staticmethod
    def setup_logger():
        if DLPipeLogger.logger is not None:
            print("WARNING: logger was setup already, deleting all previously existing handlers")
            for hdlr in DLPipeLogger.logger.handlers[:]:  # remove all old handlers
                DLPipeLogger.logger.removeHandler(hdlr)

        # Create the logger
        DLPipeLogger.logger = logging.getLogger(DLPipeLogger._logger_name)
        DLPipeLogger.logger.setLevel(DLPipeLogger._logger_level)

        # Setup the StringIO handler
        DLPipeLogger._log_contents = io.StringIO()
        DLPipeLogger.string_handler = logging.StreamHandler(DLPipeLogger._log_contents)
        DLPipeLogger.string_handler.setLevel(DLPipeLogger._logger_level)

        # Setup the console handler
        DLPipeLogger.console_handler = logging.StreamHandler()
        DLPipeLogger.console_handler.setLevel(DLPipeLogger._logger_level)

        # Setup the file handler
        DLPipeLogger.file_handler = logging.FileHandler(DLPipeLogger._current_log_file_path, 'a')
        DLPipeLogger.file_handler.setLevel(DLPipeLogger._logger_level)

        # Optionally add a formatter
        DLPipeLogger.string_handler.setFormatter(DLPipeLogger._formatter)
        DLPipeLogger.console_handler.setFormatter(DLPipeLogger._formatter)
        DLPipeLogger.file_handler.setFormatter(DLPipeLogger._formatter)

        # Add the console handler to the logger
        DLPipeLogger.logger.addHandler(DLPipeLogger.string_handler)
        DLPipeLogger.logger.addHandler(DLPipeLogger.console_handler)
        DLPipeLogger.logger.addHandler(DLPipeLogger.file_handler)

    @staticmethod
    def set_log_file(path, mode: str='a'):
        DLPipeLogger._current_log_file_path = path
        DLPipeLogger.logger.removeHandler(DLPipeLogger.file_handler)

        DLPipeLogger.file_handler = logging.FileHandler(DLPipeLogger._current_log_file_path, mode)
        DLPipeLogger.file_handler.setLevel(DLPipeLogger._logger_level)
        DLPipeLogger.logger.addHandler(DLPipeLogger.file_handler)

    @staticmethod
    def remove_file_logger():
        DLPipeLogger.logger.removeHandler(DLPipeLogger.file_handler)
        if os.path.exists(DLPipeLogger._current_log_file_path):
            os.remove(DLPipeLogger._current_log_file_path)

    @staticmethod
    def get_contents():
        return DLPipeLogger._log_contents.getvalue()

    @staticmethod
    def get_log_file_path() -> str:
        return DLPipeLogger._current_log_file_path

    @staticmethod
    def set_level(lvl):
        DLPipeLogger._logger_level = lvl
        DLPipeLogger.setup_logger()


DLPipeLogger.setup_logger()
