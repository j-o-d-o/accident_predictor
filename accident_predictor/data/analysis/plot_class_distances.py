import configparser
import numpy as np
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger

import plotly.graph_objs as go
import plotly.offline as offline


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    col = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "k_distance")

    raw_data = col.find({})

    for row in raw_data:
        y = row["distances"]
        x = np.arange(len(row["distances"]))
        title = "class_" + str(row["class"]) + "_vs_" + str(row["compared_to"])

        trace = go.Scatter(x=x, y=y)
        data = [trace]
        layout = dict(title=title)
        fig = dict(data=data, layout=layout)
        offline.plot(fig, filename='plots/' + title + '.html')
