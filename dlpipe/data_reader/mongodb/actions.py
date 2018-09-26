from dlpipe.data_reader.mongodb import MongoDBConnect
import configparser


class MongoDBActions:
    @staticmethod
    def add_config(file_name):
        cp = configparser.ConfigParser()
        if len(cp.read(file_name)) == 0:
            raise ValueError("Config File could not be loaded, please check the correct path!")
        MongoDBConnect.add_connections_from_config(cp)
