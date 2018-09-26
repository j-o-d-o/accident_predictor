import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
from accident_predictor.metrics import single_class_precision, single_class_recall
from accident_predictor.train import PreProcessData
from dlpipe.utils import DLPipeLogger
from bson import ObjectId
from keras.models import load_model
import gridfs
import os
import numpy as np
import csv


# ID of the experiment that should be loaded
EXP_ID = "5baa6dd932b90159c4fe31e9"
# Index (usually equals the epoch number + 1) of the weights that should be loaded, takes latest if None
INDEX = None


def get_class_distribution(result):
    """
    Calculate normalized class histogram
    :param result: prediction result
    :return: predicted class, normalized class histogram, difference to next best class
    """
    result_sum = sum(result)
    normalized = [
        result[0] / result_sum,
        result[1] / result_sum,
        result[2] / result_sum
    ]
    max_class = np.argmax(result)
    max_conf = normalized[int(max_class)]
    min_diff = 1.0
    for norm in normalized:
        diff = max_conf - norm
        if diff != 0 and diff < min_diff:
            min_diff = diff
    return max_class, normalized, min_diff


def print_train_data_info(row, class_prediction, distribution):
    """
    Debug function to print predicted training data for analysis
    :param row: accident data document object
    :param class_prediction: the predicted class
    :param distribution: predicted class distribution
    """
    class_actual = row["accident_severity"]
    console_str = str(class_actual) + " == " + str(class_prediction) + " | " + str(distribution) + " - "
    if int(class_prediction) == int(class_actual):
        console_str += " TRUE"
    else:
        console_str += " FALSE"
        print("")
        print(row["_id"])
    print(console_str)


if __name__ == "__main__":
    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    db = MongoDBConnect.get_db("localhost_mongo_db", "models")
    col = MongoDBConnect.get_collection("localhost_mongo_db", "models", "experiment")
    col_test = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "test")
    col_train = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")

    exp_obj = col.find_one({"_id": ObjectId(EXP_ID)})

    # load model weight data as h5 file from mongoDB
    fs = gridfs.GridFS(db)
    idx = -1 if INDEX is None else INDEX
    h5_file = fs.get(exp_obj["weights"][idx]["model_gridfs"])
    h5_bytes = h5_file.read()

    tmp_filename = "tmp_model_weights_read.h5"
    with open(tmp_filename, 'wb') as f:
        f.write(h5_bytes)

    # create model with custom metric objects as used while training
    model = load_model(tmp_filename, custom_objects={
        "p": single_class_precision(0),
        "r": single_class_recall(0),
        "p_1": single_class_precision(1),
        "r_1": single_class_recall(1),
        "p_2": single_class_precision(2),
        "r_2": single_class_recall(2),
    })
    os.remove(tmp_filename)

    data_set = col_test.find()

    csv_data = []
    class_counter = [0, 0, 0]
    processor = PreProcessData()
    for row in data_set:
        _, feat, _, _ = processor.process(row, None, None)
        feat = np.asarray([feat])
        prediction = model.predict(feat)
        class_prediction, distribution, minimum_diff = get_class_distribution(prediction[0])
        csv_data.append([row["row_id"], int(class_prediction)+1])
        class_counter[int(class_prediction)] += 1

    print("Predicted classes counter:")
    print(class_counter)

    # create result CSV file
    with open("result_" + EXP_ID + ".csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(['Unfall_ID', 'Unfallschwere'])
        for csv_row_data in csv_data:
            writer.writerow(csv_row_data)
