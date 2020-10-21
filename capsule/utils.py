

import tensorflow as tf


def squash(vectors, axis=-1):
    epsilon = 1e-8
    vector_squared_norm = tf.math.reduce_sum(tf.math.square(vectors), axis=axis, keepdims=True) + epsilon
    return (vector_squared_norm / (1 + vector_squared_norm)) * (vectors / tf.math.sqrt(vector_squared_norm)) + epsilon


def margin_loss(v_k, T_k, m_plus = 0.9, m_minus = 0.1, down_weighting = 0.5):
    L_k = T_k * tf.square(tf.maximum(0., m_plus - v_k)) + \
          down_weighting * (1. - T_k) * tf.square(tf.maximum(0., v_k - m_minus))
    
    L_k = tf.reduce_sum(L_k, axis=-1)
    return L_k