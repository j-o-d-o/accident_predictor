from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import urllib.parse
from typing import NamedTuple, List
from configparser import ConfigParser
from dlpipe.utils import DLPipeLogger


class MongoDBConnectionConfig(NamedTuple):
    name: str
    url: str
    port: int
    user: str = None
    pwd: str = None
    client: MongoClient = None


class MongoDBConnect:

    _connections: List[MongoDBConnectionConfig] = []

    @staticmethod
    def add_connection(config: MongoDBConnectionConfig):
        """
        Adds a MongoDBConnectionConfig to the connection dict
        :param config: config that should be added of type MongoDBConnectionConfig
        """
        MongoDBConnect._connections.append(config)

    @staticmethod
    def add_connections_from_config(config_parser: ConfigParser):
        """
        Takes a parsed .ini file as argument and adds all connections with type=MongoDB,
        Each section (= name) must have url, port and can have pwd and user
        :param config_parser: A ConfigParser of a .ini file
        """
        for key in config_parser.sections():
            if config_parser[key]["type"] == "MongoDB":
                MongoDBConnect.add_connection(MongoDBConnectionConfig(
                    name=key,
                    url=config_parser[key]["url"],
                    port=int(config_parser[key]["port"]),
                    pwd=config_parser[key].get("pwd"),
                    user=config_parser[key].get("user")
                ))

    @staticmethod
    def get_client(name: str) -> MongoClient:
        con, i = MongoDBConnect.get_connection_by_name(name)
        if con.client is None:
            con = MongoDBConnect.connect_to(name)
        return con.client

    @staticmethod
    def get_db(name: str, db_name: str) -> Database:
        client = MongoDBConnect.get_client(name)
        return client[db_name]

    @staticmethod
    def get_collection(name: str, db_name: str, collection: str) -> Collection:
        db = MongoDBConnect.get_db(name, db_name)
        return db[collection]

    @staticmethod
    def connect_to(name: str) -> MongoDBConnectionConfig:
        """
        Connect to connection which was previously added by its name
        :param name: Key of the connection config as string
        :return: The MongoDBConnectionConfig which the connection is to
        """
        con, i = MongoDBConnect.get_connection_by_name(name)
        DLPipeLogger.logger.info("Connect to database {0}:{1}".format(con.url, str(con.port)))

        host = con.url + ":" + str(con.port)
        if con.user is not None and con.pwd is not None:
            user = urllib.parse.quote_plus(con.user)
            pwd = urllib.parse.quote_plus(con.pwd)
            con_string = 'mongodb://%s:%s@%s' % (user, pwd, host)
        else:
            con_string = 'mongodb://%s' % host

        db_client = MongoClient(con_string)
        new_con = con._replace(client=db_client)
        MongoDBConnect._connections[i] = new_con
        return new_con

    @staticmethod
    def close_connection(name: str):
        con, i = MongoDBConnect.get_connection_by_name(name)
        if con.client is not None:
            con.client.close()
        del MongoDBConnect._connections[i]

    @staticmethod
    def get_connection_by_name(name: str) -> (MongoDBConnectionConfig, int):
        for i, con in enumerate(MongoDBConnect._connections):
            if con.name == name:
                return con, i
        raise ValueError(name + ": Connection does not exist!")

    @staticmethod
    def reset_connections():
        """ Close all added connections """
        for con in MongoDBConnect._connections:
            if con.client is not None:
                con.client.close()
        MongoDBConnect._connections = []
