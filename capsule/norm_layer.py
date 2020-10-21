

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

class Norm(tf.keras.Model):
    def call(self, inputs):
        x = tf.norm(inputs, name="norm", axis=-1)
        return x