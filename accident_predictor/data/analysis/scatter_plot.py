import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.offline as offline


DATA_INFO = {
    "age": {"norm": 98},
    "nr_person_hurt": {"norm": 3},
    "nr_vehicles": {"norm": 4}
}


def create_feature_vector(data):
    feature = [
        float(data["date"]["sin"]),
        float(data["date"]["cos"]),
        float(data["time"]["sin"]),
        float(data["time"]["cos"]),
        float(data["age"]) / DATA_INFO["age"]["norm"],
        float(data["nr_person_hurt"]) / DATA_INFO["nr_person_hurt"]["norm"],
        float(data["nr_vehicles"]) / DATA_INFO["nr_vehicles"]["norm"],
    ]
    feature += data["class"]["encoded"]
    feature += data["light"]["encoded"]
    feature += data["weather"]["encoded"]
    feature += data["ground_condition"]["encoded"]
    feature += data["gender"]["encoded"]
    feature += data["vehicle_type"]["encoded"]
    feature += data["road_type"]["encoded"]
    return feature


def create_data_set(data_set):
    # find feature vector length
    feature_length = len(create_feature_vector(raw_data[0]))

    # create feature array x of shape [n_samples, n_futures]
    x = np.zeros((raw_data.count(), feature_length))
    # create label array y of shape [n_samples] with classes [0,1,2]
    y = np.zeros(raw_data.count())

    index = 0
    for row in data_set:
        feature_list = create_feature_vector(row)
        feature_array = np.asarray(feature_list)
        x[index] = feature_array
        y[index] = int(row["accident_severity"])
        index += 1
    return x, y


def split_data(x_features, y_labels, classes=3):
    x_new = []
    y_new = []

    for i in range(classes):
        x_new.append([])
        y_new.append([])

    for i, x_feat in enumerate(x_features):
        class_index = int(x_feat["accident_severity"])
        x_new[class_index].append(x_feat)
        y_new[class_index].append(y_labels[i])

    return x_new, y_new


def plot_2d_scatter(x_features, y_labels, name):
    """
    Plat a 2D scatter plot to identify potential clusters
    :param x_features: the features are a list of 2D points
    :param y_labels: a list of class labels with ids corresponding to the x_features list
    :param name: name of the 2d scatter plot
    """
    x = [[], [], []]
    y = [[], [], []]
    for i, label in enumerate(y_labels):
        x[int(label)].append(x_features[i][0])
        y[int(label)].append(x_features[i][1])

    trace_0 = go.Scatter(x=x[0], y=y[0], name="0", mode="markers", marker=dict(size=2))
    trace_1 = go.Scatter(x=x[1], y=y[1], name="1", mode="markers", marker=dict(size=2))
    trace_2 = go.Scatter(x=x[2], y=y[2], name="2", mode="markers", marker=dict(size=3))
    data = [trace_0, trace_1, trace_2]
    layout = dict(title=name, yaxis=dict(zeroline=False), xaxis=dict(zeroline=False))
    fig = dict(data=data, layout=layout)
    offline.plot(fig, filename='plots/' + name + '.html')


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    col = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")

    # get all training data
    raw_data = col.find({})
    if raw_data.count() == 0:
        raise ValueError("Train database is empty, can not sample")

    # prepare data for the plot
    x, y = create_data_set(raw_data)

    # as the feature vector is highly dimensional, use pca to reduce to one dimension for visualization
    pca = PCA(n_components=2)
    x_fitted = pca.fit_transform(x)

    plot_2d_scatter(x_fitted, y, "feature_clusters_pca")
