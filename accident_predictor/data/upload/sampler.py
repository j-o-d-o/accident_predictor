"""
The sampler module uses the distance data calculated in "calc_class_distance.py" to sample synthetic data for classes
1 and 2 and samples the under represented classes by copying them in order to have a even class distribution.
"""
import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
from dlpipe.utils import DLPipeLogger
from accident_predictor.data.upload.data_encoder import sin_cos_representation
from accident_predictor.data.upload.calc_class_distances import upload_distances
from keras.utils.np_utils import to_categorical
import numpy as np
import copy


def generate_synth_data(col, ids, insert=True):
    """
    create synthetic data from distance calculation of entries to other classes and save to database
    :param col: collection to save synthetic data to
    :param ids: list of ids for the records that should be sampledn
    :param insert: bool to actually insert all entry (for debugging)
    :return: mongodb ids that where inserted as synthetic data
    """
    # TODO: potentially change light condition depending on what time (e.g. shouldnt be dark at 13:00)
    # TODO: same for date, chance of snow and ice in summer is rather low...

    cursor = col.find({"_id": {"$in": ids}})
    inserted_ids = []

    for row in cursor:

        # change age
        org_age = copy.deepcopy(row["age"])
        for age_idx in range(0, 18):
            age_min = 5 * age_idx
            age_max = 5 * (age_idx + 1)
            new_age = int(np.random.uniform(age_min, age_max, 1)[0])
            row["age"] = int(new_age)
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["age"] = org_age

        # change time
        org_time = copy.deepcopy(row["time"])
        for time_idx in range(0, 24):
            time_min = 60 * time_idx
            time_max = 60 * (time_idx + 1)
            new_time = int(np.random.uniform(time_min, time_max, 1)[0])
            sin_time, cos_time = sin_cos_representation(new_time, 1440)
            row["time"]["value"] = new_time
            row["time"]["sin"] = sin_time
            row["time"]["cos"] = cos_time
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["time"] = org_time

        # change date
        org_date = copy.deepcopy(row["date"])
        for date_idx in range(0, 18):
            date_min = 20 * date_idx
            date_max = 20 * (date_idx + 1)
            new_date = int(np.random.uniform(date_min, date_max, 1)[0])
            sin_date, cos_date = sin_cos_representation(new_date, 361)
            row["date"]["value"] = new_date
            row["date"]["sin"] = sin_date
            row["date"]["cos"] = cos_date
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["date"] = org_date

        # change class
        org_class = copy.deepcopy(row["class"])
        for new_index in range(0, len(org_class["encoded"])):
            row["class"] = {
                "value": "generated",
                "encoded": to_categorical(new_index, num_classes=len(org_class["encoded"])).astype(int).tolist()
            }
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["class"] = org_class

        # change weather
        org_class = copy.deepcopy(row["weather"])
        for new_index in range(0, len(org_class["encoded"])):
            row["weather"] = {
                "value": "generated",
                "encoded": to_categorical(new_index, num_classes=len(org_class["encoded"])).astype(int).tolist()
            }
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["weather"] = org_class

        # change gender
        org_class = copy.deepcopy(row["gender"])
        for new_index in range(0, len(org_class["encoded"])):
            row["gender"] = {
                "value": "generated",
                "encoded": to_categorical(new_index, num_classes=len(org_class["encoded"])).astype(int).tolist()
            }
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["gender"] = org_class

        # change vehicle type
        org_class = copy.deepcopy(row["vehicle_type"])
        for new_index in range(0, len(org_class["encoded"]) - 1):
            row["vehicle_type"] = {
                "value": "generated",
                "encoded": to_categorical(new_index, num_classes=len(org_class["encoded"])).astype(int).tolist()
            }
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["vehicle_type"] = org_class

        # change road_type
        org_class = copy.deepcopy(row["road_type"])
        for new_index in range(0, len(org_class["encoded"]) - 1):
            row["road_type"] = {
                "value": "generated",
                "encoded": to_categorical(new_index, num_classes=len(org_class["encoded"])).astype(int).tolist()
            }
            if insert:
                del row["_id"]
                inserted_ids.append(col.insert_one(row))
        row["road_type"] = org_class

    return inserted_ids


def up_sample(col, cursor, nr_create):
    """
    Sample a set amount of data by copying the existing data and saving it to mongodb
    :param col: mongodb collection where the new documents should be saved to
    :param cursor: pymongo cursor with the data that is getting sampled
    :param nr_create: how many additional documents should be created
    """
    if nr_create < 0:
        raise ValueError("Can not create negative amount of entries")

    counter = 0
    while counter < nr_create:
        for row in cursor:
            del row["_id"]
            col.insert_one(row)
            counter += 1
            if counter >= nr_create:
                break
        cursor.rewind()


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    col_train = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")
    col_distance = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "k_distance")

    # find class distances
    upload_distances()

    # get averaged class distances for class 1 and 2
    raw_distance_data_avg_class_1 = col_distance.find({"class": 1, "compared_to": {"$all": [0, 2]}})
    raw_distance_data_avg_class_2 = col_distance.find({"class": 2, "compared_to": {"$all": [0, 1]}})
    if raw_distance_data_avg_class_1.count() == 0 or raw_distance_data_avg_class_2.count() == 0:
        raise ValueError("No distance data found, need to execute 'calc_class_distance.py' first")

    # generate synthetic data from class distances
    inserted_ids_1 = generate_synth_data(col_train, raw_distance_data_avg_class_1[0]["ids"][0:70], True)
    inserted_ids_2 = generate_synth_data(col_train, raw_distance_data_avg_class_2[0]["ids"][0:20], True)

    raw_data_train_0 = col_train.find({"accident_severity": 0})
    raw_data_train_1 = col_train.find({"accident_severity": 1})
    raw_data_train_2 = col_train.find({"accident_severity": 2})

    print("Class distribution after synthetic data generation:")
    print("Class 0: " + str(raw_data_train_0.count()))
    print("Class 1: " + str(raw_data_train_1.count()))
    print("Class 2: " + str(raw_data_train_2.count()))

    # evenly sample data by copying existing data
    max_count = raw_data_train_0.count()
    up_sample(col_train, raw_data_train_1, (max_count - raw_data_train_1.count()))
    up_sample(col_train, raw_data_train_2, (max_count - raw_data_train_2.count()))
