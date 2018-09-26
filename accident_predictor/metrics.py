from keras import backend as K


def single_class_precision(interesting_class_id):
    """
    :param interesting_class_id: integer in range [0,2] to specify class
    :return: precision for the "interesting_class" -> TP / (TP + FP)
    """
    def p(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return p


def single_class_recall(interesting_class_id):
    """
    :param interesting_class_id: integer in range [0,2] to specify class
    :return: recall for the "interesting_class" -> TP / (TP + FN)
    """
    def r(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return r


def single_labels(interesting_class_id):
    """
    :param interesting_class_id: integer in range [0,2] to specify class
    :return: number of labels for the "interesting_class"
    """
    def s_l(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        return K.cast(K.maximum(K.sum(accuracy_mask), 1), 'int32')
    return s_l
