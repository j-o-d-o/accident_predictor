import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras import optimizers
from dlpipe.processors.processor_interface import IPreProcessor
from dlpipe.data_reader.mongodb import MongoDBReader, MongoDBConnect, MongoDBActions
from dlpipe.trainer import Trainer
from dlpipe.utils import DLPipeLogger
from dlpipe.callbacks import SaveExpMongoDB
from accident_predictor.metrics import single_class_precision, single_class_recall
from accident_predictor.plot_results import plot_acc_loss_graph


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


def create_data_reader(col):
    reader = MongoDBReader(
        col,
        batch_size=32,
        data_split=[78, 22, 0],  # test data is separate
        shuffle_data=True
    )
    processors = [PreProcessData()]
    reader.add_processors(processors)
    return reader


if __name__ == "__main__":
    # Prevent creation of a file to log console output
    DLPipeLogger.remove_file_logger()

    # Configure Data Reader
    MongoDBActions.add_config('./connections.ini')
    collection = MongoDBConnect.get_collection("localhost_mongo_db", "accident", "train")
    mr = create_data_reader(collection)

    # Configure Model
    inputs = Input(shape=(37,))
    x1 = Dense(1560, activation='relu')(inputs)
    x1 = Dropout(0.5)(x1)
    x2 = Dense(512, activation='relu')(x1)
    x2 = Dropout(0.4)(x2)
    x3 = Dense(64, activation='relu')(x2)
    x3 = Dropout(0.2)(x3)
    predictions = Dense(3, activation='softmax')(x3)
    model = Model(inputs=[inputs], outputs=[predictions])

    opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[
        "accuracy",
        single_class_precision(0), single_class_recall(0),
        single_class_precision(1), single_class_recall(1),
        single_class_precision(2), single_class_recall(2)
    ])

    # Train the model
    model_db = MongoDBConnect.get_db("localhost_mongo_db", "models")
    mongo_db_cb = SaveExpMongoDB(model_db, "accident_v1.0", model.get_config())
    trainer = Trainer(model=model, data_reader=mr, callbacks=[mongo_db_cb])
    trainer.train(epochs=1)

    # plot results
    exp_id = mongo_db_cb.get_exp_id()
    if exp_id is not None:
        plot_acc_loss_graph(exp_id)

    print("Experiment ID: " + str(exp_id))
    print("")
