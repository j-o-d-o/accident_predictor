import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger
from bson import ObjectId

import plotly.graph_objs as go
import plotly.offline as offline


def get_collection():
    """
    :return: collection which holds the experiments
    """
    cp = configparser.ConfigParser()
    if len(cp.read('./connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    col = MongoDBConnect.get_collection("localhost_mongo_db", "models", "experiment")
    return col


def create_plot_data(convert_data: list, batch_size: int, smooth_window: int=1):
    """
    Convert metric data into x, y graph data
    :param convert_data: list of metric objects with keys [batch, epoch, value]
    :param batch_size: maximum number of batches in one epoch
    :param smooth_window: values are averaged over the size of smooth_window
    :return: (x_values, y_values) => tuple of x,y values for the scatter plot
    """
    x_values = []
    y_values = []

    window_counter = 0
    sum_value = 0
    for i, data in enumerate(convert_data):
        sum_value += float(data["value"])

        window_counter += 1
        if window_counter == smooth_window or i == len(convert_data) - 1:
            decimal = (float(data["batch"]) / batch_size)
            x_val = float(data["epoch"]) + decimal
            x_values.append(x_val)
            y_val = sum_value / window_counter
            y_values.append(y_val)
            window_counter = 0
            sum_value = 0

    return x_values, y_values


def plot_acc_loss_graph(exp_id):
    """
    Create a scatter plot of loss and accuracy for validation and training data
    :param exp_id: Experiment Id
    """
    col = get_collection()
    exp_obj = col.find_one({"_id": ObjectId(exp_id)})

    batch_size = int(exp_obj["max_batches_per_epoch"])
    x_train_loss, y_train_loss = create_plot_data(exp_obj["metrics"]["training"]["loss"], batch_size, 50)
    x_val_loss, y_val_loss = create_plot_data(exp_obj["metrics"]["validation"]["loss"], batch_size, 1)
    x_train_acc, y_train_acc = create_plot_data(exp_obj["metrics"]["training"]["acc"], batch_size, 50)
    x_val_acc, y_val_acc = create_plot_data(exp_obj["metrics"]["validation"]["acc"], batch_size, 1)

    trace_train_loss = go.Scatter(x=x_train_loss, y=y_train_loss, mode="lines", name="training loss")
    trace_val_loss = go.Scatter(x=x_val_loss, y=y_val_loss, mode="lines", name="validation loss")
    trace_train_acc = go.Scatter(x=x_train_loss, y=y_train_acc, mode="lines", name="training accuracy")
    trace_val_acc = go.Scatter(x=x_val_loss, y=y_val_acc, mode="lines", name="validation accuracy")
    data = [trace_train_loss, trace_val_loss, trace_train_acc, trace_val_acc]
    layout = dict(title="accuracy + loss")
    fig = dict(data=data, layout=layout)
    offline.plot(fig, filename='loss_acc_' + str(exp_id) + '.html')


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    plot_exp_id = "5ba802c732b9016996d2f0cc"
    plot_acc_loss_graph(plot_exp_id)
