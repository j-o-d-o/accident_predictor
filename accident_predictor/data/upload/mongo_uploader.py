"""
Parsing the data from the csv files, encode the data in machine friendly formats and saves everything to the MongoDB.
"""
import configparser
from dlpipe.data_reader.mongodb import MongoDBConnect
import csv
from accident_predictor.data.upload.data_encoder import \
    date_encoder, class_encoder, light_encoder, ground_encoder, \
    gender_encoder, vehicle_encoder, weather_encoder, road_encoder, time_encoder
from dlpipe.utils import DLPipeLogger


FILES = {
    "train": "verkehrsunfaelle_train.csv",
    "test": "verkehrsunfaelle_test.csv"
}

if __name__ == "__main__":
    print("Start uploading")

    DLPipeLogger.remove_file_logger()

    cp = configparser.ConfigParser()
    if len(cp.read('./../../connections.ini')) == 0:
        raise ValueError("Config File could not be loaded, please check the correct path!")
    MongoDBConnect.add_connections_from_config(cp)
    collection_train = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")
    collection_test = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "test")

    print("Clean collections...")
    collection_test.delete_many({})
    collection_train.delete_many({})

    print("Upload data from csv files...")
    for mode, file_name in FILES.items():
        with open(file_name, encoding='utf-8') as file:
            data = csv.reader(file, delimiter=',')

            fields = []
            for row_nr, row_data in enumerate(data):
                if row_nr == 0:
                    # first row has all the field names
                    fields = row_data
                    fields[0] = "row_id"
                else:
                    try:
                        # convert the array to dict mapping the field names for easier readability
                        data_dict = {}
                        for i, field_name in enumerate(fields):
                            data_dict[field_name] = row_data[i]

                        # date and time are "cycle values" thus encode them to sin and cos components
                        date_value, date_sin, date_cos = date_encoder(data_dict["Unfalldatum"])
                        time_minutes, time_sin, time_cos = time_encoder(data_dict["Zeit (24h)"])

                        db_dict = {
                            "row_id": int(data_dict["row_id"]),
                            "date": {
                                "value": date_value,
                                "sin": date_sin,
                                "cos": date_cos
                            },
                            "age": int(data_dict["Alter"]),
                            "class": {
                                "value": data_dict["Unfallklasse"],
                                "encoded": class_encoder(data_dict["Unfallklasse"])
                            },
                            "light": {
                                "value": data_dict["Lichtverhältnisse"],
                                "encoded": light_encoder(data_dict["Lichtverhältnisse"])
                            },
                            "nr_person_hurt": min(3, int(data_dict["Verletzte Personen"])),
                            "nr_vehicles": min(4, int(data_dict["Anzahl Fahrzeuge"])),
                            "ground_condition": {
                                "value": data_dict["Bodenbeschaffenheit"],
                                "encoded": ground_encoder(data_dict["Bodenbeschaffenheit"])
                            },
                            "gender": {
                                "value": data_dict["Geschlecht"],
                                "encoded": gender_encoder(data_dict["Geschlecht"])
                            },
                            "time": {
                                "value": int(time_minutes),
                                "cos": time_cos,
                                "sin": time_sin
                            },
                            "vehicle_type": {
                                "value": data_dict["Fahrzeugtyp"],
                                "encoded": vehicle_encoder(data_dict["Fahrzeugtyp"])
                            },
                            "weather": {
                                "value": data_dict["Wetterlage"],
                                "encoded": weather_encoder(data_dict["Wetterlage"])
                            },
                            "road_type": {
                                "value": data_dict["Strassenklasse"],
                                "encoded": road_encoder(data_dict["Strassenklasse"])
                            }
                        }
                        if mode == "train":
                            # subtract 1 as the severities are originally [1,2,3] to map it to [0,1,2]
                            db_dict["accident_severity"] = int(data_dict["Unfallschwere"]) - 1
                            collection_train.insert_one(db_dict)

                        else:
                            collection_test.insert_one(db_dict)

                    except ValueError as err:
                        print("ValueError: " + str(err))

    print("Uploading done")
