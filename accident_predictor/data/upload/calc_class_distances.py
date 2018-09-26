"""
Call this module to calculate the "distances" between class 1->0, 1->2, 2->0 and 2->1.
This will find the entries that are the furthest apart. This distance information is needed during sampling data.
"""
import configparser
import bisect
import numba
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger

import numpy as np
from datetime import datetime


def create_feature_vector(data):
    """
    :param data: list of one mongodb row
    :return: numerical list of features
    """
    # info for normalization
    data_info = {
        "age": {"norm": 98},
        "nr_person_hurt": {"norm": 3},
        "nr_vehicles": {"norm": 4}
    }
    feature = [
        float(data["date"]["sin"]),
        float(data["date"]["cos"]),
        float(data["time"]["sin"]),
        float(data["time"]["cos"]),
        float(data["age"]) / data_info["age"]["norm"],
        float(data["nr_person_hurt"]) / data_info["nr_person_hurt"]["norm"],
        float(data["nr_vehicles"]) / data_info["nr_vehicles"]["norm"],
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
    """
    loop through raw mongodb data to create numpy arrays for easier processing further on
    :param data_set: raw data from mongodb of entries
    :return: feature vector 'x', labels 'y' and the _ids
    """
    # find feature vector length
    feature_length = len(create_feature_vector(data_set[0]))

    # create feature array x of shape [n_samples, n_futures]
    x = np.zeros((data_set.count(), feature_length))
    # create label array y of shape [n_samples] with classes [0,1,2]
    y = np.zeros(data_set.count())
    ids = []

    index = 0
    for row in data_set:
        feature_list = create_feature_vector(row)
        feature_array = np.asarray(feature_list)
        x[index] = feature_array
        y[index] = int(row["accident_severity"])
        ids.append(row["_id"])
        index += 1
    return x, y, ids


@numba.jit(nopython=True, parallel=True)
def calc_class_distance(x_main, x_compare, float_width, nr_classes):
    # the rest of the feature vector are class results, lets & compare them and subtract the nr of classes (7)
    same_class_counter = 0
    for step in range(float_width, len(x_main)):
        if x_main[step] == 1 and x_compare[step] == 1:
            same_class_counter += 1
    class_distance = nr_classes - same_class_counter
    return class_distance


@numba.jit(nopython=True, parallel=True)
def calc_float_distance(x_main, x_compare, float_width):
    # the first x values are float values, take the difference there
    result = 0
    for i in range(float_width):
        result += abs(x_main[i] - x_compare[i])
    return result


def find_top_distances(x_l, x_r, ids, k_size=None):
    """
    Return the top-k distances with ids from x_l to x_r
    :param x_l: feature vector for main class
    :param x_r: feature vector x_l of other class it is compared to
    :param ids: list with ids of x_l with same length
    :param k_size: how many top results should be stored
    :return: List of Distance instances corresponding to the x_l features vector
    """
    float_width = 7
    nr_classes = 7

    result = []
    print("Finding distances...")
    for counter, x_main in enumerate(x_l):
        integral_distance = 0
        for x_compare in x_r:
            # calculate distance from x_main to x_compare
            float_distance = calc_float_distance(x_main, x_compare, float_width)
            integral_distance += float_distance
            class_distance = calc_class_distance(x_main, x_compare, float_width, nr_classes)
            integral_distance += class_distance
        avg_distance = integral_distance / len(x_r)
        bisect.insort(result, Distance(avg_distance, ids[counter]))
        if k_size is not None and len(result) > k_size:
            del result[-1]
    return result


def match_distances(distances0: list, distances1: list):
    """
    Finds the matching distance for each _id and sort by average
    :param distances0: list of Distance objects
    :param distances1: list of Distance objects
    :return: list of Distance objects averaged
    """
    print("Match distance...")
    avg_distances = []
    hash_table_1 = create_hash_table(distances1)
    for dist in distances0:
        val_0 = dist.value
        val_1 = hash_table_1[dist._id]
        avg = (val_0 + val_1) / 2
        bisect.insort(avg_distances, Distance(avg, dist._id))
    return avg_distances


def create_hash_table(distances: list):
    """
    :param distances: list of Distance objects
    :return: dict with _ids as key and distance as value
    """
    result = {}
    for dist in distances:
        result[dist._id] = dist.value
    return result


def create_distance_data(distances: list):
    """
    :param distances: list of Distance objects
    :return: array of distance values, array of id values
    """
    ids = []
    values = []
    for dist in distances:
        ids.append(dist._id)
        values.append(dist.value)
    return values, ids


def save_to_db(col_save_to_mongodb, distance_list, class_, compared_to: list):
    val, ids = create_distance_data(distance_list)
    col_save_to_mongodb.insert_one({
        "distances": val,
        "ids": ids,
        "class": class_,
        "compared_to": compared_to,
        "created_at": datetime.now()
    })


class Distance(object):
    """
    Object to represent Distance between two entries,
    needed to use the bisect module to insert into a sorted list of Distance instances
    """
    def __init__(self, value: float=0, _id: str=""):
        self.value = value
        self._id = _id

    def __lt__(self, rh):
        # in order to have the largest values up front hack in the greater sign for lt
        return self.value > rh.value


def upload_distances():
    col = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")

    raw_data_0 = col.find({"accident_severity": 0})
    raw_data_1 = col.find({"accident_severity": 1})
    raw_data_2 = col.find({"accident_severity": 2})
    if raw_data_0.count() == 0 or raw_data_1.count() == 0 or raw_data_2.count() == 0:
        raise ValueError("At least one class does not have any samples, train database probably empty")

    x_0, y_0, id_0 = create_data_set(raw_data_0)
    x_1, y_1, id_1 = create_data_set(raw_data_1)
    x_2, y_2, id_2 = create_data_set(raw_data_2)

    print("Class 0: " + str(len(x_0)))
    print("Class 1: " + str(len(x_1)))
    print("Class 2: " + str(len(x_2)))

    # calculate "distance" for accident_severity 1 and 2 to all other accident_severities
    top_k_1_0 = find_top_distances(x_1, x_0, id_1, None)
    top_k_1_2 = find_top_distances(x_1, x_2, id_1, None)
    top_k_2_1 = find_top_distances(x_2, x_1, id_2, None)
    top_k_2_0 = find_top_distances(x_2, x_0, id_2, None)

    # as it can take a while to calc distances, the values are saved to the mongodb
    # to separate the calculation from sampling
    col_save = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "k_distance")

    print("Removing old distance data...")
    col_save.delete_many({})

    save_to_db(col_save, top_k_1_0, 1, [0])
    save_to_db(col_save, top_k_1_2, 1, [2])
    save_to_db(col_save, top_k_2_1, 2, [1])
    save_to_db(col_save, top_k_2_0, 2, [0])

    # match class distance and take average from distance to both classes, these distances ultimatly determine which
    # documents are sampled to generate synthetic data later on
    top_k_1 = match_distances(top_k_1_0, top_k_1_2)
    top_k_2 = match_distances(top_k_2_0, top_k_2_1)

    save_to_db(col_save, top_k_1, 1, [0, 2])
    save_to_db(col_save, top_k_2, 2, [0, 1])


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)

    upload_distances()
