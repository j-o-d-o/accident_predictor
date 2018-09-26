from dlpipe.processors.processor_interface import IPreProcessor
import numpy as np


DATA_INFO = {
    "age": {"norm": 98},
    "nr_person_hurt": {"norm": 3},
    "nr_vehicles": {"norm": 4}
}


class PreProcessData(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        ground_truth = np.zeros(3)
        if "accident_severity" in raw_data:
            index = min(int(raw_data["accident_severity"]), 2)
            ground_truth[index] = 1.0

        list_input = []

        # sin and cos components are already normalized
        list_input.append(float(raw_data["date"]["sin"]))
        list_input.append(float(raw_data["date"]["cos"]))
        list_input.append(float(raw_data["time"]["sin"]))
        list_input.append(float(raw_data["time"]["cos"]))

        # normalize features
        list_input.append(int(raw_data["age"]) / DATA_INFO["age"]["norm"])
        list_input.append(int(raw_data["nr_person_hurt"]) / DATA_INFO["nr_person_hurt"]["norm"])
        list_input.append(int(raw_data["nr_vehicles"]) / DATA_INFO["nr_vehicles"]["norm"])

        # some classification features have "unknown" columns at the end which are sliced off
        list_input += raw_data["class"]["encoded"]
        list_input += raw_data["light"]["encoded"]
        list_input += raw_data["weather"]["encoded"][:-1]
        list_input += raw_data["ground_condition"]["encoded"][:-1]
        list_input += raw_data["gender"]["encoded"]
        list_input += raw_data["vehicle_type"]["encoded"][:-1]
        list_input += raw_data["road_type"]["encoded"][:-1]

        input_data = np.asarray(list_input)

        return raw_data, input_data, ground_truth, piped_params
