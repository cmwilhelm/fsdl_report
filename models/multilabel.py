from typing import List

import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model

from .common import Dataset, Hyperparameters


CLASS_THRESHOLD = 0.5


def mean_iou(actual, predicted_probs):
    """
    Evaluation metric here is IOU.
    Since each example can belong to multiple classes,
    we can measure the performance on an individual example
    by:

    iou = |predicted ⋂ actual| / |predicted ⋃ actual|

    We then take the average of this value over all examples.
    """
    actual_classes = tf.cast(actual, tf.bool)
    predicted_classes = tf.cast(tf.greater(predicted_probs, CLASS_THRESHOLD), tf.bool)

    intersection = tf.reduce_sum(
        tf.cast(
            tf.math.logical_and(predicted_classes, actual_classes),
            tf.uint8
        ),
        axis=1
    )

    union = tf.reduce_sum(
        tf.cast(
            tf.math.logical_or(predicted_classes, actual_classes),
            tf.uint8
        ),
        axis=1
    )

    iou = tf.math.divide(intersection, union)

    return tf.reduce_mean(iou)


def mk_model(embedding_dim: int, num_classes: int, hyperparameters: Hyperparameters) -> Model:
    """
    Builds the model's network.

    :param embedding_dim: number of dimensions in input paper embedding
    :param num_classes: number of target classes
    :param hyperparameters: a Hyperparameters object
    """
    model = Sequential()

    for i in range(hyperparameters.num_hidden_layers):
        kwargs = {
            "activation": "relu",
            "name": "hidden{}".format(i+1)
        }

        if hyperparameters.l2_reg:
            kwargs["kernel_regularizer"] = regularizers.l2(hyperparameters.l2_reg)

        if i == 0:
            kwargs["input_shape"] = (embedding_dim,)

        model.add(Dense(hyperparameters.hidden_size, **kwargs))

    # gets a probability between 0 and 1 for each class
    model.add(Dense(num_classes, activation="sigmoid", name="output"))

    return model


def train(
        model: Model,
        train_data: Dataset,
        eval_data: Dataset,
        hyperparameters: Hyperparameters,
        num_epochs: int,
        callbacks: List[Callback]
):
    print("Compiling...")
    model.compile(
        optimizer=optimizers.Adam(hyperparameters.learning_rate),
        loss="binary_crossentropy",
        metrics=[mean_iou]
    )

    print("Fitting!")
    model.fit(
        x=train_data.inputs,
        y=train_data.labels,
        batch_size=hyperparameters.batch_size,
        validation_data=(eval_data.inputs, eval_data.labels),
        epochs=num_epochs,
        callbacks=callbacks
    )


def evaluate(model: Model, train_set: Dataset, eval_set: Dataset, test_set: Dataset):
    model.compile(metrics=[mean_iou])

    print("Train performance:")
    model.evaluate(train_set.inputs, train_set.labels)
    print("Eval performance:")
    model.evaluate(eval_set.inputs, eval_set.labels)
    print("Test performance:")
    model.evaluate(test_set.inputs, test_set.labels)

