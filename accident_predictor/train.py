import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras import optimizers, regularizers
from dlpipe.data_reader.mongodb import MongoDBReader, MongoDBConnect, MongoDBActions
from dlpipe.trainer import Trainer
from dlpipe.utils import DLPipeLogger
from dlpipe.callbacks import SaveExpMongoDB
from accident_predictor.metrics import single_class_precision, single_class_recall
from accident_predictor.plot_results import plot_acc_loss_graph
from accident_predictor.processors import PreProcessData


def create_data_reader(col):
    reader = MongoDBReader(
        col,
        batch_size=32,
        data_split=[80, 20, 0],  # test data is separate
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
    x1 = Dense(1024, activation='relu')(inputs)
    x1 = Dropout(0.4)(x1)
    x2 = Dense(512, activation='relu')(x1)
    x2 = Dropout(0.4)(x2)
    x3 = Dense(64, activation='relu')(x2)
    x3 = Dropout(0.2)(x3)
    predictions = Dense(3, activation='softmax')(x3)
    model = Model(inputs=[inputs], outputs=[predictions])

    opt = optimizers.RMSprop(lr=0.0001, decay=0.5e-6)
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
    trainer.train(epochs=30)

    # plot results
    exp_id = mongo_db_cb.get_exp_id()
    if exp_id is not None:
        col_exp = MongoDBConnect.get_collection("localhost_mongo_db", "models", "experiment")
        plot_acc_loss_graph(exp_id, col_exp)

    print("Experiment ID: " + str(exp_id))
    print("")
