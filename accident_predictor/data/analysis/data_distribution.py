"""
Plot distribution histograms for each feature by class, your browser will be spammed with plots (one for each feature)
"""
import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger

import plotly.graph_objs as go
import plotly.offline as offline
import plotly.figure_factory as ff

import numpy as np


def fill_class_field(data_obj, field_key, class_idx, field_dict):
    """
    fill the class field with its index value and if not yet set, set label for this feature class name
    :param data_obj: one document from the training data collection
    :param field_key: key of the field as string
    :param class_idx: class index as integer [0,1,2]
    :param field_dict: the dict the value and labels are added to (by reference)
    """
    index = int(np.argmax(data_obj[field_key]["encoded"]))
    field_dict[field_key]["val"][class_idx].append(index)
    for _ in range(len(field_dict[field_key]["labels"]), index + 1):
        field_dict[field_key]["labels"].append("")
    field_dict[field_key]["labels"][index] = data_obj[field_key]["value"]


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    col = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")

    # get all training data
    raw_data = col.find({})

    label_data = np.zeros(raw_data.count())
    # each field has a value, a list of 3 lists for each class and a bin_size to specify a bin size to average values
    # the keys of the dict must be the same as the keys used in the database
    fields = {
        "age": {"val": [[], [], []], "bin_size": 1},
        "nr_vehicles": {"val": [[], [], []], "bin_size": 1},
        "nr_persons_hurt": {"val": [[], [], []], "bin_size": 1},
        "time": {"val": [[], [], []], "bin_size": 60},
        "date": {"val": [[], [], []], "bin_size": 30},
    }
    # each class field has a value, a list of 3 lists for each class and a label list to name each bin
    # e.g. ["Car", "Truck", "Bike"]
    # the keys of the dict must be the same as the keys used in the database
    fields_class = {
        "road_type": {"val": [[], [], []], "labels": []},
        "weather": {"val": [[], [], []], "labels": []},
        "vehicle_type": {"val": [[], [], []], "labels": []},
        "gender": {"val": [[], [], []], "labels": []},
        "ground_condition": {"val": [[], [], []], "labels": []},
        "light": {"val": [[], [], []], "labels": []},
        "class": {"val": [[], [], []], "labels": []},
    }

    i = 0
    for row in raw_data:
        severity = int(row["accident_severity"])
        label_data[i] = severity

        # fill values for fields
        if "age" in fields:
            fields["age"]["val"][severity].append(int(row["age"]))
        if "time" in fields:
            fields["time"]["val"][severity].append(int(row["time"]["value"]))
        if "date" in fields:
            fields["date"]["val"][severity].append(int(row["date"]["value"]))
        if "nr_persons_hurt" in fields:
            fields["nr_persons_hurt"]["val"][severity].append(int(row["nr_person_hurt"]))
        if "nr_vehicles" in fields:
            fields["nr_vehicles"]["val"][severity].append(int(row["nr_vehicles"]))

        # fill values and labels for class fields
        if "road_type" in fields_class:
            fill_class_field(row, "road_type", severity, fields_class)
        if "weather" in fields_class:
            fill_class_field(row, "weather", severity, fields_class)
        if "vehicle_type" in fields_class:
            fill_class_field(row, "vehicle_type", severity, fields_class)
        if "gender" in fields_class:
            fill_class_field(row, "gender", severity, fields_class)
        if "ground_condition" in fields_class:
            fill_class_field(row, "ground_condition", severity, fields_class)
        if "light" in fields_class:
            fill_class_field(row, "light", severity, fields_class)
        if "class" in fields_class:
            fill_class_field(row, "class", severity, fields_class)

        i += 1

    # Plot the data in Histograms, one for each feature
    layout = go.Layout(bargap=0.2, bargroupgap=0.1)
    offline.plot(go.Figure(data=[go.Histogram(x=label_data)], layout=layout), filename="plots/labels.html")

    for key, data in fields.items():
        offline.plot(ff.create_distplot(
            [data["val"][0], data["val"][1], data["val"][2]],
            ["severity 0", "severity 1", "severity 2"],
            histnorm='probability density', curve_type="kde",
            show_hist=True, show_curve=True, bin_size=data["bin_size"]),
            filename=("plots/" + key + '.html'))

    for key, data in fields_class.items():
        fig = ff.create_distplot(
            [data["val"][0], data["val"][1], data["val"][2]],
            ["severity 0", "severity 1", "severity 2"],
            histnorm='probability', show_hist=True, show_curve=False, bin_size=0.99)

        fig['layout'].update(xaxis=go.layout.XAxis(tickmode="array", ticktext=data["labels"],
                                                   tickvals=[i for i in range(len(data["labels"]))]))

        offline.plot(fig, filename=("plots/" + key + '.html'))
