from typing import List

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Sequential, Model

from .common import Dataset, Hyperparameters


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

    # create probability distribution over the classes
    model.add(Dense(num_classes, activation="softmax", name="output"))

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
        loss="categorical_crossentropy",
        metrics=["accuracy"]
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
    model.compile(metrics=["accuracy"])

    print("Train performance:")
    model.evaluate(train_set.inputs, train_set.labels)
    print("Eval performance:")
    model.evaluate(eval_set.inputs, eval_set.labels)
    print("Test performance:")
    model.evaluate(test_set.inputs, test_set.labels)

