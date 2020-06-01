from typing import List, Optional
import json

import numpy as np
import tensorflow as tf


class Hyperparameters:
    def __init__(
            self,
            hidden_size: int = 64,
            num_hidden_layers: int = 2,
            l2_reg: Optional[float] = None,
            batch_size: int = 128,
            learning_rate: float = 0.01
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def __dict__(self):
        return {
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'l2_reg': self.l2_reg,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }


class Dataset:
    def __init__(
            self,
            filepath: str,
            input_attr: str,
            label_attr: str
    ):
        inputs = []
        labels = []

        with open(filepath, 'r') as f:
            for l in f:
                decoded = json.loads(l)
                inputs.append(decoded[input_attr])
                labels.append(decoded[label_attr])

        self.inputs = np.array(inputs, dtype=np.float64)
        self.labels = np.array(labels, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.inputs)


def mk_tensorboard_callback(log_dir, experiment_name):
    log_dir = log_dir + '/' + experiment_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return tensorboard_callback


def mk_checkpoint_callback(checkpoint_path):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    return cp_callback



