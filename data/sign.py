import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from random import random


def create_sign_data(batch_size, dataset_size=60000):
    train_x, train_labels = [], []
    x = 0
    while x <= 1.0:
        train_x.append([-x])  
        train_labels.append(0)
        train_x.append([+x])  
        train_labels.append(1)
        x += 1.0 / float(dataset_size / 2)

    dataset_size = len(train_x)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_labels))
    train_ds = train_ds.shuffle(dataset_size).batch(batch_size)
    return train_ds